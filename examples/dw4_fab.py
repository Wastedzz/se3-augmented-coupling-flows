import hydra
from omegaconf import DictConfig
from functools import partial



from eacf.train.train import train
from eacf.targets.data import load_dw4
from eacf.setup_run.create_fab_train_config import create_train_config
from eacf.targets.target_energy.double_well import make_dataset, log_prob_fn
from eacf.utils.data import positional_dataset_only_to_full_graph

def load_dataset_original(train_set_size: int, valid_set_size: int, final_run: bool = True):
    train, valid, test = load_dw4(train_set_size)
    if not final_run:
        return train, valid[:valid_set_size]
    else:
        return train, test[:valid_set_size]

def load_dataset_custom(batch_size, train_set_size: int = 1000, test_set_size:int = 1000, seed: int = 0,
                        temperature: float = 1.0):
    dataset = make_dataset(seed=seed, n_vertices=4, dim=2, n_samples=test_set_size+train_set_size,
                           temperature=temperature)

    train_set = dataset[:train_set_size]
    train_set = train_set[:train_set_size - (train_set.shape[0] % batch_size)]

    test_set = dataset[-test_set_size:]
    return positional_dataset_only_to_full_graph(train_set), positional_dataset_only_to_full_graph(test_set)


def to_local_config(cfg: DictConfig) -> DictConfig:
    """Change config to make it fast to run locally. Also remove saving."""
    # Training
    cfg.training.optimizer.init_lr = 1e-4
    cfg.training.batch_size = 16
    cfg.training.n_epoch = int(1e3) + 23
    cfg.training.save = True
    cfg.training.n_eval = 10
    cfg.training.plot_batch_size = 32
    cfg.training.K_marginal_log_lik = 2
    cfg.fab.eval_inner_batch_size = 32
    cfg.fab.eval_total_batch_size = 64
    cfg.fab.buffer_min_length_batches = cfg.fab.n_updates_per_smc_forward_pass
    cfg.fab.buffer_max_length_batches = cfg.fab.n_updates_per_smc_forward_pass*10
    cfg.logger = DictConfig({"list_logger": None})
    # cfg.logger = DictConfig({"pandas_logger": {'save_period': 50}})

    # Flow
    cfg.flow.type = 'spherical'
    cfg.flow.n_aug = 1
    cfg.flow.n_layers = 1
    cfg.training.resume = False


    # Configure NNs
    cfg.flow.nets.mlp_head_config.mlp_units = (4,)
    cfg.flow.nets.egnn.mlp_units = (4,)
    cfg.flow.nets.egnn.n_blocks = 2
    cfg.flow.nets.non_equivariant_transformer_config.output_dim = 3
    cfg.flow.nets.non_equivariant_transformer_config.mlp_units = (4,)
    cfg.flow.nets.non_equivariant_transformer_config.n_layers = 2
    cfg.flow.nets.non_equivariant_transformer_config.num_heads = 1
    cfg.training.use_multiple_devices = True

    debug = False
    if debug:
        cfg_train = dict(cfg['training'])
        cfg_train['debug'] = True
        cfg.training = DictConfig(cfg_train)
    return cfg


@hydra.main(config_path="./config", config_name="dw4_fab.yaml")
def run(cfg: DictConfig):
    local_config = True
    if local_config:
        print("running locally")
        cfg = to_local_config(cfg)

    print(cfg)
    if cfg.target.custom_samples:
        print(f"loading custom dataset for temperature of {cfg.target.temperature}")
        load_dataset = partial(load_dataset_custom, temperature=cfg.target.temperature)
    else:
        load_dataset = load_dataset_original
    experiment_config = create_train_config(cfg, target_log_p_x_fn=log_prob_fn,
                                            dim=2, n_nodes=4, load_dataset=load_dataset,
                                            date_folder=True)
    print(experiment_config)
    train(experiment_config)


if __name__ == '__main__':
    run()
