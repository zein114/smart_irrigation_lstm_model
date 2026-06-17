import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import config

class ToleranceAccuracy(tf.keras.metrics.Metric):
    def __init__(self, tolerance=0.05, name='tolerance_accuracy', **kwargs):
        super(ToleranceAccuracy, self).__init__(name=name, **kwargs)
        self.tolerance = tolerance
        self.correct_predictions = self.add_weight(name='cp', initializer='zeros')
        self.total_predictions = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        is_within_tolerance = tf.less_equal(tf.abs(y_true - y_pred), self.tolerance)
        is_within_tolerance = tf.cast(is_within_tolerance, tf.float32)
        self.correct_predictions.assign_add(tf.reduce_sum(is_within_tolerance))
        self.total_predictions.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return (self.correct_predictions / self.total_predictions) * 100.0

    def reset_state(self):
        self.correct_predictions.assign(0.0)
        self.total_predictions.assign(0.0)

class RSquare(tf.keras.metrics.Metric):
    def __init__(self, name='r_square', **kwargs):
        super(RSquare, self).__init__(name=name, **kwargs)
        self.sum_squared_errors = self.add_weight(name='sse', initializer='zeros')
        self.sum_total_errors = self.add_weight(name='sst', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        sse = tf.reduce_sum(tf.square(y_true - y_pred))
        sst = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        self.sum_squared_errors.assign_add(sse)
        self.sum_total_errors.assign_add(sst)

    def result(self):
        return 1.0 - (self.sum_squared_errors / (self.sum_total_errors + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.sum_squared_errors.assign(0.0)
        self.sum_total_errors.assign(0.0)

def build_lstm_model(input_shape):
    # Architecture améliorée pour mieux gérer les données non-séquentielles
    model = Sequential([
        Input(shape=input_shape),
        
        # Première couche LSTM avec plus d'unités
        LSTM(64, activation='tanh', return_sequences=True),
        Dropout(0.3),
        
        # Deuxième couche LSTM
        LSTM(32, activation='tanh', return_sequences=False),
        Dropout(0.3),
        
        # Couche Dense supplémentaire pour extraire plus de features
        Dense(16, activation='relu'),
        Dropout(0.2),
        
        # Couche de sortie
        Dense(1, activation='linear')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='huber',
        metrics=['mae', ToleranceAccuracy(), RSquare()]
    )
    return model