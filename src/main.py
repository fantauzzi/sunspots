from pathlib import Path
import hydra
from omegaconf import DictConfig
import mlflow
from sunspots.utils import hydra_config_file_name, hydra_config_file_path


# mlflow run . -P steps='all' -P overrides='dataset.path=data'

@hydra.main(version_base=None, config_path=hydra_config_file_path, config_name=hydra_config_file_name)
def main(config: DictConfig):
    available_steps = ['fetch', 'preprocess', 'train', 'test']
    required_steps = available_steps if config['main']['steps'] == 'all' else config['main']['steps'].split(',')

    if 'fetch' in required_steps:
        mlflow.run(uri=str(Path(hydra.utils.get_original_cwd()) / 'fetch_dataset'),
                   entry_point='main',
                   parameters={'url': config['dataset']['url'],
                               'path': config['dataset']['path']})
    if 'preprocess' in required_steps:
        mlflow.run(uri=str(Path(hydra.utils.get_original_cwd()) / 'preprocessing'),
                   entry_point='main',
                   parameters={'input_file': str(Path(config['dataset']['path']) / 'SN_m_tot_V2.0.csv'),
                               'train_size': 0.8,
                               'train_file': str(Path(config['dataset']['path']) / 'train.csv'),
                               'test_file': str(Path(config['dataset']['path']) / 'test.csv')})
    if 'train' in required_steps:
        mlflow.run(uri=str(Path(hydra.utils.get_original_cwd()) / 'model_training'),
                   entry_point='main',
                   parameters={'train_file': str(Path(config['dataset']['path']) / 'train.csv'),
                               'validation_size': 0.33333,
                               'model_file': str(Path(config['dataset']['path']) / 'trained_tf_model')})
    if 'test' in required_steps:
        mlflow.run(uri=str(Path(hydra.utils.get_original_cwd()) / 'model_test'),
                   entry_point='main',
                   parameters={'model_file': str(Path(config['dataset']['path']) / 'trained_tf_model'),
                               'test_file': str(Path(config['dataset']['path']) / 'test.csv')})


if __name__ == '__main__':
    main()
