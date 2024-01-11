import glob
import os
from datetime import datetime
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import joblib

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Concatenate, TimeDistributed, Conv1D, \
    MaxPooling1D, Flatten, Bidirectional, RepeatVector, Reshape, \
    Dropout, BatchNormalization, LayerNormalization
from keras.layers import Bidirectional
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from attention import Attention


def normalize_data(train_X, train_y):
    X_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))

    # X_scaler = RobustScaler()
    # y_scaler = RobustScaler()

    train_X = train_X.reshape(train_X.shape[0], -1)

    train_X_scaled = X_scaler.fit_transform(train_X)
    train_y_scaled = y_scaler.fit_transform(train_y)

    # Exporting Scalers
    joblib.dump(X_scaler, "../model/LSTM_Scaler/X_scaler.pkl")
    joblib.dump(y_scaler, "../model/LSTM_Scaler/y_scaler.pkl")

    return train_X_scaled, train_y_scaled, X_scaler, y_scaler


def fit_lstm_model(train_X, train_y, n_batch, n_epoch, n_neurons, val=0.05):
    # Define the model
    ts_input = Input((train_X.shape[1], train_X.shape[2]))
    mask_layer = tf.keras.layers.Masking(mask_value=-1, )(ts_input)

    lstm1_1_out = LSTM(int(n_neurons), activation="tanh", return_sequences=True)(mask_layer)
    lstm1_1_lynorm = LayerNormalization()(lstm1_1_out)
    dropout_1 = Dropout(0.2)(lstm1_1_lynorm)

    lstm1_2_out = LSTM(int(n_neurons), activation="tanh", return_sequences=True)(dropout_1)
    lstm1_2_lynorm = LayerNormalization()(lstm1_2_out)
    dropout_2 = Dropout(0.2)(lstm1_2_lynorm)

    lstm1_3_out = LSTM(int(n_neurons), activation="tanh", return_sequences=True)(dropout_2)
    lstm1_3_lynorm = LayerNormalization()(lstm1_3_out)
    dropout_3 = Dropout(0.2)(lstm1_3_lynorm)

    # lstm1_4_out = LSTM(int(n_neurons), activation="tanh", return_sequences=True)(dropout_3)
    # lstm1_4_lynorm = LayerNormalization()(lstm1_4_out)
    # dropout_4 = Dropout(0.2)(lstm1_4_lynorm)

    attention_layer = Attention(units=512, name='Custom_Attention')(dropout_3)
    outputs = Dense(1)(attention_layer)

    # intialize & compile
    model = Model(inputs=ts_input, outputs=outputs)

    # Set Learning rate decay for AdamOpt
    #     initial_lr = 0.0002 # Default for adam is 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.80,
        staircase=True
    )

    Adam_Opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # # Compile the model
    # model.compile(optimizer=Adam_Opt, loss=q_loss.call)
    # Huber Loss
    model.compile(optimizer=Adam_Opt, loss=tf.keras.losses.Huber())
    # model.compile(optimizer=Adam_Opt, loss='mae')

    print(f'Model Output shape: {y.shape}')

    # patient early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    # Set up Tensorboard
    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f'Tensorboard log path: {logdir}')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    history = model.fit(train_X, train_y, epochs=n_epoch,
                        batch_size=n_batch,
                        verbose=1, shuffle=True,
                        validation_split=val,
                        # validation_data=([val_X1, val_X2], val_y),
                        callbacks=[tensorboard_callback, es])

    return model


def make_predictions(model, n_batch, inputs):
    predictions = model.predict(inputs, batch_size=n_batch)

    return predictions


def evaluate_forecasts(truth, pred):
    r_square = r2_score(truth, pred)
    mae = mean_absolute_error(truth, pred)
    mape = np.mean(np.abs((truth - pred) / truth)) * 100
    rmse = np.sqrt(mean_squared_error(truth, pred))
    mbe = np.mean(pred - truth)

    print(f"R2: {r_square}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}")
    print(f"RMSE: {rmse}")
    print(f"MBE: {mbe}")
    print('==============================')

    return None


if __name__ == "__main__":
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    print(os.getenv('TF_GPU_ALLOCATOR'))
    # Set GPU Memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    '''
    # Load Training Data
    X_path = glob.glob("../data/input_TS/TS_X_*.npy")
    y_path = glob.glob("../data/input_TS/TS_y_*.npy")

    print(f"Num. of Days of Training Set: {len(X_path)}")
    
    X = np.concatenate([np.load(X_path) for X_path in X_path])
    y = np.concatenate([np.load(y_path) for y_path in y_path])
    '''

    X = np.load('../data/input_TS_merge/TS_X_all.npy')
    y = np.load('../data/input_TS_merge/TS_y_all.npy')
    # # Adjust for inputs
    # # dist_mask = np.array(X[:, :, 15] <= 500 * 1000, dtype=int)
    # # dist_mask[dist_mask == 0] = -1
    # # X[:, :, 15] = dist_mask
    # # X[:, :, 16] = X[:, :, 16] * dist_mask
    # # X[X[:, :, 16] == 0] = -1
    mask = list(range(0, 22))
    mask.remove(15)
    X = X[:, :, mask]

    # X = np.load('../data/input_TS_merge/TS_X_filter.npy')
    # y = np.load('../data/input_TS_merge/TS_y_filter.npy')

    # Only keep non-negative records
    X = X[(y > 0).squeeze(), :, :]
    y = y[(y > 0).squeeze(), :]
    # Keep records without NaNs
    nonan_idx = np.argwhere(~np.isnan(X).any(axis=(1, 2))).squeeze()
    X = X[nonan_idx, :, :]
    y = y[nonan_idx, :]

    time_lag = X.shape[1]
    n_features = X.shape[2]

    print(f"Total Training Samples: {X.shape}, {y.shape}")

    # Normalize Inputs
    X, y, X_scaler, y_scaler = normalize_data(X, y)
    # Replace NaN with -1
    X = np.nan_to_num(X, nan=-1)
    # Reshape X to LSTM format
    X = X.reshape(-1, time_lag, n_features)

    # Create a KFold object
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    fold_num = 1
    # Iterate over the folds
    for train_idx, val_idx in kfold.split(X):
        train_X = X[train_idx]
        train_y = y[train_idx]

        val_X = X[val_idx]
        val_y = y[val_idx]

        print(f"Fold Num: {fold_num} Train Size: {train_X.shape} | Test Size: {val_X.shape}")

        fold_num += 1

        # Set GPU Memory
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        model = fit_lstm_model(train_X=train_X, train_y=train_y,
                               n_batch=2 ** 7, n_epoch=20,
                               n_neurons=2 ** 9, val=0.05)

        print("Model Training Completed!")
        # Calling `save('my_model.h5')` creates a h5 file `my_model.h5`.
        model.save(f"../model/LSTM_model/LSTM_{fold_num}.h5")
        print('Model Saved!')

        K.clear_session()

        # Evaluate Train
        # print("============= Train =================")
        # pred_train = make_predictions(model=model, n_batch=2 ** 7, inputs=train_X)
        #
        # print('Inverse forecast to unscaled numbers!')
        #
        # inv_pred_train = y_scaler.inverse_transform(pred_train)
        # inv_y = y_scaler.inverse_transform(train_y)
        #
        # evaluate_forecasts(truth=inv_y, pred=inv_pred_train)

        # Evaluate Test
        print("============= Test =================")
        pred_val = make_predictions(model=model, n_batch=2 ** 8, inputs=val_X)

        print('Inverse forecast to unscaled numbers!')

        inv_pred_val = y_scaler.inverse_transform(pred_val)
        inv_y_val = y_scaler.inverse_transform(val_y)

        evaluate_forecasts(truth=inv_y_val, pred=inv_pred_val)
        # evaluate_forecasts(truth=val_y, pred=pred_val)

        print("Training Finished!")
