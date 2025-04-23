import hydra
from omegaconf import DictConfig
from functools import partial
import jax
import pickle
from eacf.setup_run.create_train_config import create_flow_config
from eacf.flow.build_flow import build_flow
from eacf.train.train import train
from eacf.train.fab_generate import fab_eval_function
from eacf.targets.data import load_lj55
from eacf.setup_run.create_fab_train_config import create_train_config
from eacf.targets.target_energy.double_well import make_dataset, log_prob_fn
from eacf.utils.data import positional_dataset_only_to_full_graph
import torch
import numpy as np
import os

def load_dataset(train_set_size: int, valid_set_size: int, final_run: bool = True):
    train, valid, test = load_lj55(train_set_size)
    if not final_run:
        return train, valid[:valid_set_size]
    else:
        return train, test[:valid_set_size]

def to_local_config(cfg: DictConfig) -> DictConfig:
    """Change config to make it fast to run locally. Also remove saving."""
    cfg.flow.nets.type = "egnn"
    cfg.flow.nets.egnn.mlp_units = (4,)
    cfg.flow.n_layers = 1
    cfg.flow.nets.egnn.n_blocks = 2
    cfg.training.batch_size = 2
    cfg.flow.type = 'spherical'
    cfg.flow.n_aug = 1
    cfg.fab.eval_inner_batch_size = 2
    cfg.fab.eval_total_batch_size = 4
    cfg.fab.n_updates_per_smc_forward_pass = 2
    cfg.fab.n_intermediate_distributions = 4
    cfg.fab.buffer_min_length_batches = 4
    cfg.fab.buffer_max_length_batches = 10

    cfg.training.n_epoch = 50
    cfg.training.save = True
    cfg.training.resume = True
    cfg.training.plot_batch_size = 4
    cfg.logger = DictConfig({"list_logger": None})

    debug = False
    if debug:
        cfg_train = dict(cfg['training'])
        cfg_train['scan_run'] = False
        cfg.training = DictConfig(cfg_train)

    return cfg


@hydra.main(config_path="./config", config_name="lj55_fab.yaml")
def run(cfg: DictConfig):
    local_config = False
    if local_config:
        print("running locally")
        cfg = to_local_config(cfg)
    
    key = jax.random.PRNGKey(0)
    flow_config = create_flow_config(cfg)
    flow = build_flow(flow_config)
    state_dir = cfg.fab.ckpt_path
    with open(state_dir, "rb") as f:
        state = pickle.load(f)
    train_data, test_data = load_dataset(cfg.training.train_set_size, cfg.training.test_set_size)
    flow_samples = fab_eval_function(
        state=state, key=key, flow=flow,
        log_p_x=log_prob_fn,
        features=test_data.features[0],
        batch_size=cfg.fab.eval_total_batch_size,
        inner_batch_size=cfg.fab.eval_inner_batch_size
        )
    flow_samples = torch.tensor(np.asarray(flow_samples))
    os.makedirs('./fab_results/{}/'.format(cfg.fab.save_des),exist_ok=True)
    torch.save(flow_samples, './fab_results/{}/samples_{}.pt'.format(cfg.fab.save_des, cfg.fab.eval_total_batch_size))


if __name__ == '__main__':
    run()
