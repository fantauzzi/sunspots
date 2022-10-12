# Credits: model taken from Coursera's "Sequences, Time Series and Prediction" course by DeepLearning.AI
import pandas as pd
import numpy as np
import tensorflow as tf
import click
from pathlib import Path


# mlflow run . -P input_file="../../data/SN_m_tot_V2.0.csv"

@click.command()
@click.option('--train_file',
              required=True,
              help='File with the dataset for training/validation/test',
              type=str)
@click.option('--validation_size',
              required=True,
              help='''Size of the validation set. If less than 1, it is meant to be the fraction of the training set to 
              be used for validation; if at least 1, it is meant to be the number of samples from the training set to be
              used for validation instead''',
              type=float)
@click.option('--model_file',
              required=True,
              help='Output file where the trained model will be saved.',
              type=str)
def main(train_file: str, validation_size: float, model_file: str) -> None:
    path_to_prj_root = '../..'
    train_file_corrected = str(path_to_prj_root / Path(train_file))
    df = pd.read_csv(train_file_corrected)
    dataset_size = len(df)
    time = np.arange(0, dataset_size, 1)
    series = df['mean'].to_numpy()

    # Split the dataset
    val_set_size = int(validation_size) if validation_size >= 1 else int(validation_size * dataset_size)
    train_set_size = dataset_size - val_set_size
    time_train = time[:train_set_size]
    x_train = series[:train_set_size]
    time_valid = time[train_set_size:]
    x_valid = series[train_set_size:]
    assert len(x_train) == len(time_train) == train_set_size
    assert len(x_valid) == len(time_valid) == val_set_size
    """time_test = time[-test_set_size:]
    x_test = series[-test_set_size:]
    assert len(time_test) == len(x_test)"""

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
    epochs = 100

    # Generate the dataset windows
    train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

    # Build the model
    # Build the Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                               strides=1,
                               activation="relu",
                               padding='causal',
                               input_shape=[window_size, 1]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 400)
    ])

    # Print the model summary
    model.summary()

    # Set the learning rate
    learning_rate = 8e-7

    # Set the optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    # Set the training parameters
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])

    # Train the model
    history = model.fit(train_set, epochs=epochs)

    # Get mae and loss from history log
    mae = history.history['mae']
    loss = history.history['loss']

    model_file_corrected = str(path_to_prj_root / Path(model_file))
    model.save(model_file_corrected)


if __name__ == '__main__':
    main()
