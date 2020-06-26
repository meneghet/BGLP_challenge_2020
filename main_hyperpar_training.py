import pandas as pd
import numpy as np
import os
import itertools as iter
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

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

    # ====== Building results dictionary
    results = dict()
    results['val_loss'] = list()
    results['to_use'] = list()
    results['split_prc'] = list()
    
    # ====== Building hyperparameter dictionary
    hyperparameter = dict()

    # define feature sets to test using power set of list--------------------
    feat_set = ['DR','pie','pce','ic_pe','exercise_ob_2nd']

    def pset(lst):
        comb = (iter.combinations(lst, l) for l in range(1, len(lst) + 1))
        return list(iter.chain.from_iterable(comb))

    power_feat_set = pset(feat_set)
    
    # add CGM in every feature set
    for n, el in enumerate(power_feat_set):
        power_feat_set[n] = ('CGM',) + power_feat_set[n]
    
    hyperparameter['to_use'] = list()
    for f in power_feat_set:
        hyperparameter['to_use'].append(f)
    hyperparameter['to_use'].append(('CGM',))
    
    # split for validation set
    hyperparameter['split_prc'] = list()
    hyperparameter['split_prc'].append(0.5)
    hyperparameter['split_prc'].append(0.6)
    hyperparameter['split_prc'].append(0.7)
    hyperparameter['split_prc'].append(0.8)
        
    iteration = 1
    n_iteration = len(hyperparameter['to_use']) * len(hyperparameter['split_prc'])
    n_rep = 3
    
    lr = 0.001
        
    start = time.time()
    for split_prc in hyperparameter['split_prc']:
        for to_use in hyperparameter['to_use']:
            val_loss_list = list()
            print(F"Iteration: {iteration}/{n_iteration}, Feats: {'-'.join(to_use)}, SPLT: {split_prc}")
            
            for rep in np.arange(n_rep):
                model, val_loss = fit_model(X_tr,Y_tr,PH = PH,patient=patient,lookback = lookback,
                                            is_multi_output = is_multi_output,to_use = list(to_use),split_prc = split_prc, lr = lr,
                                            needs_to_be_trained=True, verbose = False, is_model_saved = False)
                val_loss_list.append(val_loss)
                
            results['val_loss'].append(np.mean(val_loss_list))				
            results['to_use'].append(to_use)
            results['split_prc'].append(split_prc)

            iteration = iteration + 1
            end = time.time()
            print(F"Elapsed: {(end-start)/60/60:.2f} hours")

    # ====== Save the results
    if is_multi_output:
        out_str = 'multi_output'
    else:
        out_str = 'single_output'
    save_dir = os.path.join('results','hpar_full_selection')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    fname = F'results_training.{patient}.ph{PH}.{out_str}.npy'
    np.save(os.path.join(save_dir,fname),results)
    

def fit_model(X_tr,Y_tr,PH,patient,lookback,is_multi_output,to_use,split_prc,lr,
    needs_to_be_trained = True,is_model_saved = True,verbose = True):

    # ====== Delete nans and memorize their idx
    X_tr, Y_tr, idx_nan_tr = remove_nan_features(X_tr,Y_tr,to_use)
    start_islands, end_islands = find_nan_islands(X_tr)

    # ====== Scale features
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_tr.loc[:,:] = scaler_X.fit_transform(X_tr)

    Y_tr.loc[:] = scaler_Y.fit_transform(Y_tr)

    #Prepare data for LSTM
    split_idx = round(split_prc*X_tr.shape[0])

    train_data = X_tr.iloc[0:split_idx,:].copy()
    validation_data = X_tr.iloc[split_idx:,:].copy()
    training_full_data = X_tr.copy()

    
    train_data['Y'] = Y_tr.Y[0:split_idx]
    validation_data['Y'] = Y_tr.Y[split_idx:]
    training_full_data['Y'] = Y_tr.Y

    n_steps = int(lookback/5)

    if(is_multi_output):
        n_outputs = int(PH/5)
    else:
        n_outputs = 1
    
    X_train,Y_train,removed_idx_train = split_sequences_multi_output_correct_nan(train_data,n_steps=n_steps,n_outputs=n_outputs, start_islands=start_islands,end_islands=end_islands)
    X_val,Y_val,removed_idx_val = split_sequences_multi_output_correct_nan(validation_data,n_steps=n_steps, n_outputs=n_outputs, start_islands=start_islands,end_islands=end_islands)
    X_train_full,Y_train_full,removed_idx_train_full = split_sequences_multi_output_correct_nan(training_full_data,n_steps=n_steps, n_outputs=n_outputs, start_islands=start_islands,end_islands=end_islands)

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
            if not os.path.exists(os.path.join('results','model_weights')):
                os.mkdir(os.path.join('results','model_weights'))
            checkpoint = ModelCheckpoint(os.path.join('results','model_weights','weights.'+patient+'.ph'+str(PH)+'.splitprc'+str(split_prc)+'.lr'+str(lr)+'.feats'+'-'.join(to_use)+'.'+out_str+'.best.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            history = model.fit(X_train,Y_train,validation_data = (X_val,Y_val), epochs = 1000, batch_size = batch_size, 
                        callbacks=[early_stop,rmse_monitor,checkpoint],
                        shuffle = True)
        else:
            history = model.fit(X_train,Y_train,validation_data = (X_val,Y_val), epochs = 1000, batch_size = batch_size, 
                        callbacks=[early_stop,rmse_monitor],
                        shuffle = True, verbose = verbose)
        #Plot history
        loss = history.history['loss']
        val_loss = history.history['val_loss']
    
    if is_model_saved:
        model.load_weights(os.path.join('results','model_weights','weights.'+patient+'.ph'+str(PH)+'.splitprc'+str(split_prc)+'.lr'+str(lr)+'.feats'+'-'.join(to_use)+'.'+out_str+'.best.hdf5'))

    return model, np.sqrt(min(val_loss))*np.sqrt(scaler_Y.var_)


if __name__ == '__main__':
    main()
