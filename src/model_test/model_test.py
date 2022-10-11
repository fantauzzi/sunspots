# Credits: model taken from Coursera's "Sequences, Time Series and Prediction" course by DeepLearning.AI
import pandas as pd
import numpy as np
import tensorflow as tf
import click
from pathlib import Path


# mlflow run . -P model_file="data/trained_tf_model" -P test_file="data/test.csv"

@click.command()
@click.option('--model_file',
              required=True,
              help='Input file with the model to be tested.',
              type=str)
@click.option('--test_file',
              required=True,
              help='File with the dataset for model testing',
              type=str)
def main(model_file: str, test_file: str) -> None:
    path_to_prj_root = '../..'
    model_file_corrected = str(path_to_prj_root / Path(model_file))
    model = tf.keras.models.load_model(model_file_corrected)
    test_file_corrected = str(path_to_prj_root / Path(test_file))
    df = pd.read_csv(test_file_corrected)
    dataset_size = len(df)
    time_test = np.arange(0, dataset_size, 1)
    x_test = df['mean'].to_numpy()
    assert len(time_test) == len(x_test)

    # Prepare features and labels
    def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
        """Generates dataset windows

        Args:
          series (array of float) - contains the values of the time series
          window_size (int) - the number of time steps to include in the feature
          batch_size (int) - the batch size
          shuffle_buffer(int) - buffer size to use for the shuffle method

        Returns:
          dataset (TF Dataset) - TF Dataset containing time windows
        """

        # Generate a TF Dataset from the series values
        dataset = tf.data.Dataset.from_tensor_slices(series)

        # Window the data but only take those with the specified size
        dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

        # Flatten the windows by putting its elements in a single batch
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

        # Create tuples with features and labels
        dataset = dataset.map(lambda window: (window[:-1], window[-1]))

        # Shuffle the windows
        dataset = dataset.shuffle(shuffle_buffer)

        # Create batches of windows
        dataset = dataset.batch(batch_size).prefetch(1)

        return dataset

    # Parameters
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000
    epochs = 10

    # Print the model summary
    model.summary()

    # Get number of epochs
    # epochs = range(len(loss))

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

    forecast_series = x_test[:-1]
    forecast = model_forecast(model, forecast_series, window_size, batch_size)
    results = forecast.squeeze()
    test_mae = tf.keras.metrics.mean_absolute_error(x_test[window_size:], results).numpy()
    print(f'Test MAE is {test_mae}')


if __name__ == '__main__':
    main()
