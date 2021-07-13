import utils as util
import tensorflow as tf
import numpy as np
def forecast_model(series, time,forecastDays):
    split_time=2555
    time_train=time[:split_time]
    x_train=series[:split_time]
    split_time_test=3285
    time_valid=time[split_time:split_time_test]
    x_valid=series[split_time:split_time_test]
    time_test=time[split_time_test:]
    x_test=series[split_time_test:]

    window_size=30
    batch_size=32
    shuffle_buffer_size=1000

    tf.keras.backend.clear_session()
    tf.random.set_seed(51)
    np.random.seed(51)
    train_set = util.windowed_dataset(x_train, window_size=60, batch_size=100, shuffle_buffer=shuffle_buffer_size)
    valid_set=util.windowed_dataset(x_valid,window_size,batch_size,shuffle_buffer_size)
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=60, kernel_size=5,
                        strides=1, padding="causal",
                        activation="relu",
                        input_shape=[None, 1]),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400)
    ])

    optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                optimizer=optimizer,
                metrics=["mae"])
    history = model.fit(train_set,validation_data=(valid_set),epochs=5)

    rnn_forecast = util.model_forecast(model, series[..., np.newaxis], window_size)
    rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
    mae=tf.keras.metrics.mean_absolute_error(x_test, rnn_forecast[:365]).numpy()
    accuracy=100-mae
    return (accuracy,mae,rnn_forecast[:forecastDays])