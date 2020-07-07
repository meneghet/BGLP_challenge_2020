import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns 
import os
import time
import pdb
import argparse
import warnings
warnings.filterwarnings("ignore")

# ====== Models 
#Sklearn models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
#Keras
from keras.layers import Dense,Dropout, SimpleRNN, GRU,LSTM, Masking, Bidirectional,Conv1D,MaxPooling1D
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop, Adam
import keras.backend as K

# ======  Selection 
from sklearn.model_selection import GridSearchCV

# ====== Preprocessing 
from sklearn.preprocessing import StandardScaler

# ======  Plots 
from keras.utils import plot_model

# ======  User-defined libs 
#Evaluation metrics
from metrics import *
#Utilities
from utils import *
#Graphics
#from graphics import *
#Model building 
from model_building import *


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('patN', type=int)
    parser.add_argument('PH', type=int)
    args = parser.parse_args()
    patN = args.patN
    PH = args.PH
    
    #PH = 30
    #patN = 540
    
    # ====== Read data
    patient = str(patN)
    data_tr = pd.read_csv('data/ohio'+patient+'_Training.txt',sep=',')
    data_te = pd.read_csv('data/ohio'+patient+'_Testing.txt',sep=',')
    
    # ====== Pre-selection of features 
    is_multi_output = False
    lookback = 15
    if is_multi_output:
        distance_target = 5
    else:
        distance_target = PH
    X_tr, Y_tr = preselection(data_tr,PH=distance_target)
    X_te, Y_te = preselection(data_te,PH=distance_target)
    X_tr = X_tr.drop(columns=['Time'])
    X_te = X_te.drop(columns=['Time'])

    # ====== Delete nan targets and memorize their idx 
    X_tr, X_te, Y_tr, Y_te, idx_nan_Y_tr, idx_nan_Y_te = remove_nan_target(X_tr,X_te,Y_tr,Y_te)

    # ====== Load training results
    if is_multi_output:
        out_str = 'multi_output'
    else:
        out_str = 'single_output'
    results = np.load(os.path.join('results','hpar_full_selection','results_training.'+patient+'.ph'+str(PH)+'.'+out_str+'.npy'),allow_pickle='TRUE').item()

    # ====== Find the best try
    best_try = np.where(results['val_loss'] == np.min(results['val_loss']))[0][0]
    hyper_best = {
        'split_prc' : results['split_prc'][best_try],
        'to_use' : results['to_use'][best_try]
    }
    hyper_best['to_use'] = list(hyper_best['to_use'])
    
    print("Training best model : (Feats: " + "-".join(hyper_best['to_use']) + ", SPLT: "+ str(hyper_best['split_prc']) + ")")

    for rep in np.arange(5):

        print('Repetition {}/{}'.format(rep+1,5))
        # ====== Train the best model and get predictions
        model, val_loss, Y_tr_lstm, Y_te_lstm, X_tr_lstm, X_te_lstm = fit_and_test_model(X_tr,X_te,Y_tr,Y_te,PH = PH, patient = patient, lookback = lookback,
                        is_multi_output = is_multi_output,
                        to_use = hyper_best['to_use'],
                        split_prc = hyper_best['split_prc'],
                        lr = 0.01,
                        needs_to_be_trained=True, 
                        verbose = False, 
                        is_model_saved = False)
        
        # ====== Put prediction in the right position and rebuild tr-te 
        #Add the lookback sample used by the lstm 

        n_steps = int(lookback/5)
        Y_tr_lstm_reb = np.concatenate((np.full([n_steps-1], np.nan),Y_tr_lstm))
        Y_te_lstm_reb = np.concatenate((np.full([n_steps-1], np.nan),Y_te_lstm))
        
        #Put back the nans...
        Y_tr_pred = pd.DataFrame(index = data_tr.index)
        Y_te_pred = pd.DataFrame(index = data_te.index)
        
        #...and fill with the predicitons
        Y_tr_pred['time'] = data_tr.Time
        Y_tr_pred['prediction'] = np.full([Y_tr_pred.shape[0],1], np.nan)
        Y_tr_pred.prediction[X_tr_lstm.index] = Y_tr_lstm_reb
        
        Y_te_pred['time'] = data_te.Time
        Y_te_pred['prediction'] = np.full([Y_te_pred.shape[0],1], np.nan)
        Y_te_pred.prediction[X_te_lstm.index] = Y_te_lstm_reb
        
        # ====== Save predictions to .csv
        save_dir = os.path.join('results','test_final_pred')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        Y_tr_pred.to_csv(os.path.join(save_dir,F'subj{patient}_PH{PH}_training_rep{rep+1}.csv'),index=False,na_rep='NA')
        Y_te_pred.to_csv(os.path.join(save_dir,F'subj{patient}_PH{PH}_test_rep{rep+1}.csv'),index=False,na_rep='NA')

def fit_and_test_model(X_tr,X_te,Y_tr,Y_te,PH,patient,lookback,is_multi_output,to_use,split_prc,lr,
    needs_to_be_trained = True,is_model_saved = True,verbose = True):

    # ====== Delete nans and memorize their idx
    X_tr, Y_tr, idx_nan_tr = remove_nan_features(X_tr,Y_tr,to_use)
    start_islands_tr, end_islands_tr = find_nan_islands(X_tr)

    X_te, Y_te, idx_nan_te = remove_nan_features(X_te,Y_te,to_use)
    start_islands_te, end_islands_te = find_nan_islands(X_te)


    # ====== Scale features
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_tr.loc[:,:] = scaler_X.fit_transform(X_tr)
    X_te.loc[:,:] = scaler_X.transform(X_te)

    Y_tr.loc[:] = scaler_Y.fit_transform(Y_tr)
    Y_te.loc[:] = scaler_Y.transform(Y_te)

    #Prepare data for LSTM
    split_idx = round(split_prc*X_tr.shape[0])

    train_data = X_tr.iloc[0:split_idx,:].copy()
    validation_data = X_tr.iloc[split_idx:,:].copy()
    test_data = X_te.copy()
    training_full_data = X_tr.copy()

    
    train_data['Y'] = Y_tr.Y[0:split_idx]
    validation_data['Y'] = Y_tr.Y[split_idx:]
    test_data['Y'] = Y_te.Y
    training_full_data['Y'] = Y_tr.Y

    n_steps = int(lookback/5)

    if(is_multi_output):
        n_outputs = int(PH/5)
    else:
        n_outputs = 1
    
    X_train,Y_train,removed_idx_train = split_sequences_multi_output_correct_nan(train_data,n_steps=n_steps,n_outputs=n_outputs, start_islands=start_islands_tr,end_islands=end_islands_tr)
    X_val,Y_val,removed_idx_val = split_sequences_multi_output_correct_nan(validation_data,n_steps=n_steps, n_outputs=n_outputs, start_islands=start_islands_tr,end_islands=end_islands_tr)
    X_train_full,Y_train_full,removed_idx_train_full = split_sequences_multi_output_correct_nan(training_full_data,n_steps=n_steps, n_outputs=n_outputs, start_islands=start_islands_tr,end_islands=end_islands_tr)
    X_test,Y_test,removed_idx_test = split_sequences_multi_output_correct_nan(test_data,n_steps=n_steps, n_outputs=n_outputs, start_islands=start_islands_te,end_islands=end_islands_te)

    # ====== Model building
    model = build_model(X_train,n_outputs = n_outputs,lr=lr, verbose = verbose)
    
    # ====== Set weight file name substring
    if is_multi_output:
        out_str = 'multi_output'
    else:
        out_str = 'single_output'

    if needs_to_be_trained:
    
        batch_size = 128 #almost 17 using 256

        class RMSEMonitor(Callback):
            def on_epoch_end(self, batch, logs={}):
                if verbose:
                    print(" --> RMSE: " + str(np.sqrt(float(logs.get('val_loss')))*np.sqrt(scaler_Y.var_)) + " mg/dL")
        rmse_monitor = RMSEMonitor()
        early_stop = EarlyStopping(monitor='val_loss', mode='min',verbose=verbose, patience = 25)

        if is_model_saved:
            save_dir = os.path.join('results','model_weights')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            checkpoint = ModelCheckpoint(os.path.join(save_dir,'weights.'+patient+'.ph'+str(PH)+'.splitprc'+str(split_prc)+'.lr'+str(lr)+'.feats'+'-'.join(to_use)+'.'+out_str+'.best.hdf5'),
                        monitor='val_loss', verbose=verbose, save_best_only=True, mode='min')
            history = model.fit(X_train,Y_train,validation_data = (X_val,Y_val), epochs = 1000, batch_size = batch_size, 
                        callbacks=[early_stop,rmse_monitor,checkpoint],
                        shuffle = True, verbose = verbose)
        else:
            history = model.fit(X_train,Y_train,validation_data = (X_val,Y_val), epochs = 1000, batch_size = batch_size, 
                        callbacks=[early_stop,rmse_monitor],
                        shuffle = True, verbose = verbose)
        #Plot history
        loss = history.history['loss']
        val_loss = history.history['val_loss']
    
    if is_model_saved:
        model.load_weights(os.path.join('results','model_weights','weights.'+patient+'.ph'+str(PH)+'.splitprc'+str(split_prc)+'.lr'+str(lr)+'.feats'+'-'.join(to_use)+'.'+out_str+'.best.hdf5'))
    
    Y_te_lstm = model.predict(X_test)[:,-1]*np.sqrt(scaler_Y.var_)+scaler_Y.mean_
    Y_tr_lstm = model.predict(X_train_full)[:,-1]*np.sqrt(scaler_Y.var_)+scaler_Y.mean_
    print("Train RMSE: " + str(root_mean_squared_error(Y_train_full[:,-1]*np.sqrt(scaler_Y.var_)+scaler_Y.mean_,Y_tr_lstm)))
    print("Test RMSE: " + str(root_mean_squared_error(Y_test[:,-1]*np.sqrt(scaler_Y.var_)+scaler_Y.mean_,Y_te_lstm)))

    return model, np.sqrt(min(val_loss))*np.sqrt(scaler_Y.var_), Y_tr_lstm, Y_te_lstm, cleanse_X(X_tr,n_steps,end_islands_tr), cleanse_X(X_te,n_steps,end_islands_te)

if __name__ == '__main__':
    main()