from typing import Union, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from fabjax.sampling.smc import SequentialMonteCarloSampler
from fabjax.sampling.resampling import log_effective_sample_size

from eacf.flow.aug_flow_dist import AugmentedFlow, GraphFeatures
from eacf.train.fab_train_no_buffer import TrainStateNoBuffer, flat_log_prob_components, get_joint_log_prob_target
from eacf.train.fab_train_with_buffer import TrainStateWithBuffer


def fab_eval_function(state: Union[TrainStateNoBuffer, TrainStateWithBuffer],
                      key: chex.PRNGKey,
                      flow: AugmentedFlow,
                      smc: SequentialMonteCarloSampler,
                      log_p_x,
                      features: GraphFeatures,
                      batch_size: int,
                      inner_batch_size: int) -> dict:
    """Evaluate the ESS of the flow, and AIS. """
    assert smc.alpha == 1.  # Make sure target_energy is set to p.
    assert smc.use_resampling is False  # Make sure we are doing AIS, not SMC.

    # Setup scan function.
    features_with_multiplicity = features[:, None]
    n_nodes = features.shape[0]
    event_shape = (n_nodes, 1 + flow.n_augmented, flow.dim_x)
    features_with_multiplicity = features[:, None]
    flatten, unflatten, log_p_flat_fn, log_q_flat_fn, flow_log_prob_apply, flow_log_prob_apply_with_extra = \
        flat_log_prob_components(
        log_p_x=log_p_x, flow=flow, params=state.params, features_with_multiplicity=features_with_multiplicity,
        event_shape=event_shape
    )

    def inner_fn(carry: None, xs: chex.PRNGKey) -> Tuple[None, Tuple[chex.Array, chex.Array]]:
        """Perform SMC forward pass and grab just the importance weights."""
        key = xs
        sample_flow, log_q_flow = flow.sample_and_log_prob_apply(state.params, features, key, (inner_batch_size,))
        x0 = flatten(sample_flow.positions)
        return x0 

    # Run scan function.
    n_batches = int(np.ceil(batch_size // inner_batch_size))
    flow_sample = jax.lax.scan(inner_fn, init=None, xs=jax.random.split(key, n_batches))

    return flow_sample
