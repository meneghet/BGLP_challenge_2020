from keras.layers import Dense,Dropout, SimpleRNN, GRU,LSTM, Masking, Bidirectional,Conv1D,MaxPooling1D
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop, Adam
import keras.backend as K
from keras.utils import plot_model

def build_model(X_train,n_outputs,lr=0.001,verbose = True):
    
    #Model building
    dropout = 0.1
    recurrent_dropout = 0.2

    model = Sequential()
    model.add(Bidirectional(LSTM(128,
                return_sequences=True,
                dropout = dropout,
                recurrent_dropout=recurrent_dropout,
                stateful=False),
                input_shape = X_train[0].shape
                #batch_size=batch_size
            ))
    model.add(LSTM(64, return_sequences=True,
                dropout = dropout,
                recurrent_dropout=recurrent_dropout,
                stateful=False,
                ))
    model.add(LSTM(32,
                dropout = dropout,
                recurrent_dropout=recurrent_dropout,
                stateful = False))
    model.add(Dense(n_outputs, activation='linear'))

    optimizer = RMSprop(lr = lr)
    model.compile(optimizer=optimizer, loss='mse')
    
    if verbose:
        model.summary()

    return model