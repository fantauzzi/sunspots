# Credits: model taken from Coursera's "Sequences, Time Series and Prediction" course by DeepLearning.AI

import pandas as pd
import numpy as np
import tensorflow as tf
from utils import get_silso_dataset_info


def main():
    _, dataset_filepath = get_silso_dataset_info()
    df = pd.read_csv(dataset_filepath)
    df = df[(df.n_observations > 0) & (df.provisional == 'n')]
    dataset_size = len(df)
    time = np.arange(0, dataset_size - 1, 1)
    series = df['mean'].to_numpy()

    # Split the dataset
    train_val_test_ratios = (3, 1, 1)
    train_set_size = int(train_val_test_ratios[0] / sum(train_val_test_ratios) * dataset_size)
    val_set_size = int(train_val_test_ratios[1] / sum(train_val_test_ratios) * dataset_size)
    test_set_size = dataset_size - train_set_size - val_set_size
    time_train = time[:train_set_size]
    x_train = series[:train_set_size]
    time_valid = time[train_set_size: train_set_size + val_set_size]
    x_valid = series[train_set_size: train_set_size + val_set_size]
    assert len(x_train) == len(time_train) == train_set_size
    assert len(x_valid) == len(time_valid) == val_set_size
    time_test = time[-test_set_size:]
    x_test = series[-test_set_size:]
    assert len(time_test) == len(x_test)
    pass

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

    # Get number of epochs
    epochs = range(len(loss))

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

    forecast_series = series[train_set_size + val_set_size :-1]
    forecast = model_forecast(model, forecast_series, window_size, batch_size)
    results = forecast.squeeze()
    test_mae = tf.keras.metrics.mean_absolute_error(x_test[window_size:], results).numpy()
    print(f'Test MAE is {test_mae}')







if __name__ == '__main__':
    main()
