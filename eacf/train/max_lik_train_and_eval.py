from typing import Callable, Tuple, Optional

import chex
import jax
import jax.numpy as jnp

from eacf.utils.test import random_rotate_translate_permute

from eacf.utils.testing import get_checks_for_flow_properties
from eacf.flow.distrax_with_extra import Extra
from eacf.flow.aug_flow_dist import AugmentedFlow, FullGraphSample, AugmentedFlowParams
from eacf.train.base import maybe_masked_mean

Params = chex.ArrayTree
X = chex.Array
LogProbWithExtraFn = Callable[[Params, X], Tuple[chex.Array, Extra]]


def masked_ml_loss_fn(
        key: chex.PRNGKey,
        params: AugmentedFlowParams,
        x: FullGraphSample,
        verbose_info: bool,
        flow: AugmentedFlow,
        use_flow_aux_loss: bool,
        aux_loss_weight: float,
        apply_random_rotation: bool = False,
        log_q_cutoff: float = -1e4  # Ignore points outside realm of numerical stability.
) -> Tuple[chex.Array, Tuple[chex.Array, dict]]:
    if apply_random_rotation:
        key, subkey = jax.random.split(key)
        rotated_positions = random_rotate_translate_permute(x.positions[:, None], subkey, translate=False, permute=False)
        rotated_positions = jnp.squeeze(rotated_positions, axis=1)
        x = x._replace(positions=rotated_positions)
    aux_samples = flow.aux_target_sample_n_apply(params.aux_target, x, key)
    joint_samples = flow.separate_samples_to_joint(x.features, x.positions, aux_samples)
    log_q, extra = flow.log_prob_with_extra_apply(params, joint_samples)
    # Train by maximum likelihood.
    loss = - log_q
    info = {"mean_log_prob_q_joint": log_q,
            }
    aux_loss = jnp.mean(extra.aux_loss)
    info.update(flow.get_base_and_target_info(params))
    if use_flow_aux_loss:
        loss = loss + aux_loss * aux_loss_weight
    if verbose_info:
        info.update({"layer_info/" + key: value for key, value in extra.aux_info.items()})
    info.update(aux_loss=aux_loss)

    mask = jnp.isfinite(loss) & (log_q > log_q_cutoff)
    return loss, (mask, info)


def general_ml_loss_fn(
        key: chex.PRNGKey,
        params: AugmentedFlowParams,
        x: FullGraphSample,
        verbose_info: bool,
        flow: AugmentedFlow,
        use_flow_aux_loss: bool,
        aux_loss_weight: float,
        apply_random_rotation: bool = False
) -> Tuple[chex.Array, dict]:
    if apply_random_rotation:
        key, subkey = jax.random.split(key)
        rotated_positions = random_rotate_translate_permute(x.positions, subkey, translate=False, permute=False)
        x = x._replace(positions=rotated_positions)
    aux_samples = flow.aux_target_sample_n_apply(params.aux_target, x, key)
    joint_samples = flow.separate_samples_to_joint(x.features, x.positions, aux_samples)
    log_q, extra = flow.log_prob_with_extra_apply(params, joint_samples)
    mean_log_prob_q = jnp.mean(log_q)
    # Train by maximum likelihood.
    loss = - mean_log_prob_q
    info = {"mean_log_prob_q_joint": mean_log_prob_q,
            }
    aux_loss = jnp.mean(extra.aux_loss)
    info.update(flow.get_base_and_target_info(params))
    if use_flow_aux_loss:
        loss = loss + aux_loss * aux_loss_weight
    if verbose_info:
        info.update({"layer_info/" + key: value for key, value in extra.aux_info.items()})
    info.update(aux_loss=aux_loss)
    return loss, info


def get_tree_leaf_norm_info(tree):
    norms = jax.tree_util.tree_map(jnp.linalg.norm, tree)
    norms = jnp.stack(jax.tree_util.tree_flatten(norms)[0])
    max_norm = jnp.max(norms)
    min_norm = jnp.min(norms)
    mean_norm = jnp.mean(norms)
    median_norm = jnp.median(norms)
    info = {}
    info.update(per_layer_max_norm=max_norm, per_layer_min_norm=min_norm,
                per_layer_mean_norm=mean_norm, per_layer_median_norm=median_norm)
    return info


def get_eval_on_test_batch(params: AugmentedFlowParams,
                           x_test: FullGraphSample,
                           key: chex.PRNGKey,
                           flow: AugmentedFlow,
                           K: int,
                           test_invariances: bool = True,
                           mask: Optional[chex.Array] = None) -> dict:

    key, subkey = jax.random.split(key)
    x_augmented, log_p_a = flow.aux_target_sample_n_and_log_prob_apply(params.aux_target, x_test, subkey, K)
    x_test = jax.tree.map(lambda x: jnp.repeat(x[None, ...], K, axis=0), x_test)
    joint_sample = flow.separate_samples_to_joint(x_test.features, x_test.positions, x_augmented)

    log_q = jax.vmap(flow.log_prob_apply, in_axes=(None, 0))(params, joint_sample)
    chex.assert_equal_shape((log_p_a, log_q))
    log_w = log_q - log_p_a

    info = {}
    info.update(eval_log_lik=maybe_masked_mean(jnp.mean(log_q, axis=0), mask=mask))
    marginal_log_lik = maybe_masked_mean(jax.nn.logsumexp(log_w, axis=0) - jnp.log(jnp.array(K)),
                                         mask=mask)
    info.update(marginal_log_lik=marginal_log_lik)

    lower_bound_marginal_gap = marginal_log_lik - maybe_masked_mean(jnp.mean(log_w, axis=0), mask=mask)
    info.update(lower_bound_marginal_gap=lower_bound_marginal_gap)

    info.update(var_log_w=maybe_masked_mean(jnp.var(log_w, axis=0), mask=mask))
    info.update(ess_aug_conditional=maybe_masked_mean(
        1 / jnp.sum(jax.nn.softmax(log_w, axis=0) ** 2, axis=0) / log_w.shape[0], mask=mask))

    if test_invariances:
        key, subkey = jax.random.split(key)
        invariances_info = get_checks_for_flow_properties(joint_sample[0], flow=flow, params=params, key=subkey,
                                                          mask=mask)
        info.update(invariances_info)
    return info

def get_eval_on_test_batch_with_further(
        params: AugmentedFlowParams,
        x_test: FullGraphSample,
        key: chex.PRNGKey,
        flow: AugmentedFlow,
        K: int,
        target_log_prob: Optional[Callable] = None,
        test_invariances: bool = True,
        mask: Optional[chex.Array] = None,
        ) -> Tuple[chex.Array, dict]:
    # First copy and paste `get_eval_on_test_batch` function. Then additionally return log_w
    key, subkey = jax.random.split(key)
    x_augmented, log_p_a = flow.aux_target_sample_n_and_log_prob_apply(params.aux_target, x_test, subkey, K)
    x_test = jax.tree.map(lambda x: jnp.repeat(x[None, ...], K, axis=0), x_test)
    joint_sample = flow.separate_samples_to_joint(x_test.features, x_test.positions, x_augmented)

    log_q = jax.vmap(flow.log_prob_apply, in_axes=(None, 0))(params, joint_sample)
    chex.assert_equal_shape((log_p_a, log_q))
    log_w = log_q - log_p_a

    info = {}
    info.update(eval_log_lik=maybe_masked_mean(jnp.mean(log_q, axis=0), mask=mask))
    marginal_log_lik = maybe_masked_mean(jax.nn.logsumexp(log_w, axis=0) - jnp.log(jnp.array(K)),
                                         mask=mask)
    info.update(marginal_log_lik=marginal_log_lik)

    lower_bound_marginal_gap = marginal_log_lik - maybe_masked_mean(jnp.mean(log_w, axis=0), mask=mask)
    info.update(lower_bound_marginal_gap=lower_bound_marginal_gap)

    info.update(var_log_w=maybe_masked_mean(jnp.var(log_w, axis=0), mask=mask))

    if test_invariances:
        key, subkey = jax.random.split(key)
        invariances_info = get_checks_for_flow_properties(joint_sample[0], flow=flow, params=params, key=subkey,
                                                          mask=mask)
        info.update(invariances_info)

    if target_log_prob is None:
        log_w_joint = log_w[0]
    else:
        log_w_joint = target_log_prob(x_test.positions[0]) - log_w[0]
    return log_w_joint, info


def eval_non_batched(params: AugmentedFlowParams, single_feature: chex.Array,
                     key: chex.PRNGKey, flow: AugmentedFlow,
                     n_samples: int,
                     inner_batch_size: int,
                     target_log_prob: Callable = None,
                     target_log_prob_traceable: bool = True):

    def forward(carry, key: chex.PRNGKey):
        joint_x_flow, log_prob_flow = flow.sample_and_log_prob_apply(params, single_feature, key, (inner_batch_size,))
        features, x_positions, a_positions = flow.joint_to_separate_samples(joint_x_flow)
        log_p_x = target_log_prob(x_positions) if target_log_prob is not None and target_log_prob_traceable else None
        log_p_a_given_x = flow.aux_target_log_prob_apply(params.aux_target,
                                       FullGraphSample(features=features, positions=x_positions), a_positions)
        return None, (x_positions, a_positions, log_prob_flow, log_p_x, log_p_a_given_x)

    n_batches = int(n_samples // inner_batch_size) + 1

    _, result = jax.lax.scan(forward, None, xs=jax.random.split(key, n_batches), length=n_batches)
    result = jax.tree.map(lambda x: jnp.reshape(x, (x.shape[0]*x.shape[1], *x.shape[2:])), result)
    x_positions, a_positions, log_prob_flow, log_p_x, log_p_a_given_x = result  # unpack.

    info = {}
    original_centre = jnp.mean(x_positions, axis=-2)
    aug_centre = jnp.mean(a_positions[:, :, 0, :], axis=-2)
    info.update(mean_aug_orig_norm=jnp.mean(jnp.linalg.norm(original_centre-aug_centre, axis=-1)))

    if target_log_prob is not None:
        if not target_log_prob_traceable:
            log_p_x = target_log_prob(x_positions)
        # Calculate reverse ESS.
        log_w = log_p_x + log_p_a_given_x - log_prob_flow
        ess = 1 / jnp.sum(jax.nn.softmax(log_w) ** 2) / log_w.shape[0]
        info.update(
            {
             "rv_ess": ess}
        )
    return info


def calculate_forward_ess(log_w: chex.Array, mask: chex.Array) -> dict:
    """Calculate forward ess.
    log_w = p(x)/q(x), where x ~ p(x).
    This can be passed as the `further_fn` to `eacf.train.base.eval_fn`."""
    chex.assert_equal_shape((log_w, mask))
    log_w = jnp.where(mask, log_w, jnp.zeros_like(log_w))  # make sure log_w finite
    log_z_inv = jax.nn.logsumexp(-log_w, b=mask) - jnp.log(jnp.sum(mask))
    log_z_expectation_p_over_q = jax.nn.logsumexp(log_w, b=mask) - jnp.log(jnp.sum(mask))
    log_forward_ess = - log_z_inv - log_z_expectation_p_over_q
    forward_ess = jnp.exp(log_forward_ess)
    info = {}
    info.update(forward_ess=forward_ess)
    return info


