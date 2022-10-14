import tensorflow as tf
from pathlib import Path
import hydra
from hydra.core.global_hydra import GlobalHydra

hydra_config_file_path = Path.cwd() / '../../config'
hydra_config_file_name = 'config.yaml'


def init_hydra():
    if not GlobalHydra().is_initialized():
        hydra.initialize_config_dir(config_dir=str(hydra_config_file_path), version_base=None)
    params = hydra.compose(config_name=hydra_config_file_name)
    return params


def model_forecast(model, series, window_size, batch_size):
    """Uses an input model to generate predictions on data windows

    Args:
      model (TF Keras Model) - model that accepts data windows
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the window
      batch_size (int) - the batch size

    Returns:
      forecast (numpy array) - array containing predictions
    """

    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda w: w.batch(window_size))

    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)

    # Get predictions on the entire dataset
    forecast = model.predict(dataset)

    return forecast
