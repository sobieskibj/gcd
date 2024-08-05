import wandb
import hydra
import logging
import omegaconf
from omegaconf import DictConfig

from utils import extract_output_dir_path_from_config, set_seed

log = logging.getLogger(__name__)

def pre_run(config: DictConfig):
    set_seed(config.seed)
    output_dir = extract_output_dir_path_from_config(config)
    config.output_dir = output_dir

def wandb_setup(config: DictConfig):
    wandb_config = omegaconf.OmegaConf.to_container(
        config, resolve = True, throw_on_missing = True)
    if config.mode == 'MULTIRUN':
        suffix = f'_{config.output_dir.parts[-1]}'
        wandb_name = config.wandb.name + suffix
    else:
        wandb_name = config.wandb.name
    wandb.init(
        project = config.wandb.project,
        group = config.wandb.group,
        name = wandb_name,
        config = wandb_config)

def post_run():
    wandb.finish()

@hydra.main(version_base = None, config_path = '../configs', config_name = 'config')
def main(config: DictConfig) -> None:
    pre_run(config)
    wandb_setup(config)
    strategy = hydra.utils.instantiate(config.strategy)
    strategy.run()
    post_run()

if __name__ == '__main__':
    main()