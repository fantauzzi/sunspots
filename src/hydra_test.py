import hydra
from omegaconf import DictConfig, OmegaConf
import os


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f'Working dir is {os.getcwd()}')
    print(f'Config:\n{OmegaConf.to_yaml(cfg)}')


if __name__ == '__main__':
    main()
