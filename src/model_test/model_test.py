import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import click
from sunspots.utils import model_forecast


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

    # Parameters
    window_size = 30
    batch_size = 32

    # Print the model summary
    model.summary()

    forecast_series = x_test[:-1]
    forecast = model_forecast(model, forecast_series, window_size, batch_size)
    results = forecast.squeeze()
    test_mae = tf.keras.metrics.mean_absolute_error(x_test[window_size:], results).numpy()
    print(f'Test MAE is {test_mae}')


if __name__ == '__main__':
    main()
