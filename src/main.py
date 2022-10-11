from pathlib import Path
import hydra
from omegaconf import DictConfig
import mlflow


# mlflow run . -P steps='all' -P overrides='dataset.path=data'

@hydra.main(version_base=None, config_path='.', config_name='config.yaml')
def main(config: DictConfig):
    print(f'Configuration ===================\n{config}')

    available_steps = ['fetch', 'train']
    required_steps = available_steps if config['main']['steps'] == 'all' else config['main']['steps'].split(',')

    if 'fetch' in required_steps:
        mlflow.run(uri=str(Path(hydra.utils.get_original_cwd()) / 'fetch_dataset'),
                   entry_point='main',
                   parameters={'url': config['dataset']['url'],
                               'path': config['dataset']['path']})
    if 'train' in required_steps:
        mlflow.run(uri=str(Path(hydra.utils.get_original_cwd()) / 'model_training'),
                   entry_point='main',
                   parameters={'input_file': str(Path(config['dataset']['path']) / 'SN_m_tot_V2.0.csv')})


if __name__ == '__main__':
    main()
