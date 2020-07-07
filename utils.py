import numpy as np 
import pandas as pd
import pdb
def train_test_split(X, Y, train_size = 0.5, shuffle = True, random_state = 1):
    
    #X = np.array(X)
    #Y = np.array(Y)

    idx = int(train_size * X.shape[0]) + 1

    if shuffle:
        perm = np.random.RandomState(seed=random_state).permutation(X.shape[0])
        X, Y = X.iloc[perm], Y.iloc[perm]

    X_train, X_test = np.split(X, [idx])
    Y_train, Y_test = np.split(Y, [idx])

    #perm = np.random.RandomState(seed=random_state).permutation(X_train.shape[0])
    #X_train, Y_train = X_train.iloc[perm], Y_train.iloc[perm]
    
    return X_train, X_test, Y_train, Y_test

def create_data_prophet(X_train, X_test, Y_train, Y_test):
    data_train = pd.DataFrame()
    data_test = pd.DataFrame()

    data_train['ds']=X_train.Time
    data_train['y']=X_train.CGM
    
    #TODO: add other regressors 

    data_test['ds']=X_test.Time
    data_test['y']=X_test.CGM
    
    return data_train, data_test

    
def preselection(data,
                use_CGM = True,
                use_insulin = True,
                use_cho = True,
                use_gsr = True,
                use_temperature = True,
                use_sleep = True,
                use_exercise = True,
                use_stressors = False,
                use_work = False, 
                use_illness = False,
                use_lollo = True,
                PH = 30,
                remove_nan_target = True):

    #Drop these CGM features for the moment
    #data = data.drop(columns=['hour_of_day'])
    data = data.drop(columns=['isHyper','hypo_event'])
    data = data.drop(columns=['gfm'])
    data = data.drop(columns=['ifm'])
    data = data.drop(columns=['SR'])
    data = data.drop(columns=['t_h180','t_h250'])
    data = data.drop(columns=['gxi'])
    data = data.drop(columns=['gxc'])
    
    
    if not use_insulin:
        data = data.drop(columns=['pie'])
        data = data.drop(columns=['IOB'])
        data = data.drop(columns=['ic_OB'])
        data = data.drop(columns=['ic_pe'])
        
    #Excluded: they are impulsive
    data = data.drop(columns=['basal_insulin','bolus_insulin'])
    data = data.drop(columns=['ic','ic_w_bolus'])    
    
    if not use_cho:
        data = data.drop(columns=['COB'])
        data = data.drop(columns=['pce'])
        
    #Excluded: they are impulsive
    data = data.drop(columns=['CHO'])
    data = data.drop(columns=['is_breakfast','is_lunch','is_dinner','is_hypotreatment','is_snack'])     

    if (not use_cho or not use_insulin) or not use_lollo:
        data = data.drop(columns=['icob','dcob'])

    if not use_exercise:
        data = data.drop(columns=['exercise_ob_1st'])
        data = data.drop(columns=['exercise_ob_2nd'])
    #Excluded: they are impulsive 
    data = data.drop(columns=['exercise'])

    if not use_sleep:
        data = data.drop(columns=['sleep'])
        
    if not use_gsr:
        data = data.drop(columns=['basis_gsr'])
        
    if not use_temperature:
        data = data.drop(columns=['basis_skin_temperature'])
        
    if not use_stressors:
        data = data.drop(columns=['stressors'])
        
    if not use_work:
        data = data.drop(columns=['work'])
        
    if not use_illness:
        data = data.drop(columns=['illness'])

    #Compute target
    prediction_strategy = 'direct' #'cyclic', 'direct'

    if prediction_strategy == 'cyclic':
        steps = 1
    elif prediction_strategy == 'direct':
        steps = int(PH/5)
        
    Y = pd.DataFrame()
    Y['Y'] = data['CGM'].iloc[steps:]
    Y.index = data.index[:-steps]
    
    #X = data.iloc[:-steps]
    X = data 
    data['Y'] = Y

    Y = pd.DataFrame()
    Y['Y'] = data['Y']
    Y.index = data.index

    X = X.drop(columns=['Y'])

    if not use_CGM:
        X = X.drop(columns=['CGM','DR','der','slope'])
    return X, Y

#https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/


def remove_nan_target(X_train, X_test, Y_train, Y_test):

    idx = np.where(~np.isnan(Y_train))[0]
    idx_nan_tr = np.where(np.isnan(Y_train))[0]
    X_train = X_train.iloc[idx,:]
    Y_train = Y_train.iloc[idx]

    idx = np.where(~np.isnan(Y_test))[0]
    idx_nan_te = np.where(np.isnan(Y_test))[0]
    X_test = X_test.iloc[idx,:]
    Y_test = Y_test.iloc[idx]

    return X_train, X_test, Y_train, Y_test, idx_nan_tr, idx_nan_te

def remove_nan_features(X, Y, to_use):

    #Select only some features
    X = X[to_use]

    #Delete rows with nans
    idx_nan = X.isnull().any(axis=1)
    idx = ~X.isnull().any(axis=1)
    X = X[idx]
    Y = Y[idx]

    return X, Y, np.where(idx_nan)[0]

def find_nan_islands(data):
    start_island = list()
    end_island = list()
    
    idx = data.index[0]

    for i in np.arange(data.shape[0]-1)+1:
        if(not (data.index[i] == idx+1)):

            end_island.append(data.index[i]-1)
            start_island.append(idx+1)
            
        idx = data.index[i]
    
    #if(idx_nan.shape[0]>0):
    #    end_island.append(idx_nan[i])

    return start_island, end_island

#Univariate
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def split_sequences_multi_output(sequences, n_steps, n_outputs):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if (end_ix+n_outputs-1) > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[(end_ix-1):(end_ix-1+n_outputs), -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def split_sequences_multi_output_correct_nan(data,n_steps,n_outputs, start_islands, end_islands):

    #1. Expand start_islands and end_islands to obtain an array defining non valid indeces
    non_valid_indices = list()

    for si in start_islands:
        for i in np.arange(n_steps-1):
            non_valid_indices.append(si-(1+i))

    #for ei in end_islands:
    #    for i in np.arange(n_steps-1):
    #        non_valid_indices.append(ei+1+i)

    sequences = np.array(data)
    X, y, removed_idx = list(), list(), list()
    for i in range(len(sequences)):

        if(not np.isin(data.index[i],non_valid_indices)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if (end_ix+n_outputs-1) > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[(end_ix-1):(end_ix-1+n_outputs), -1]
            X.append(seq_x)
            y.append(seq_y)

        else:
            removed_idx.append(i)

    return np.array(X), np.array(y), np.array(removed_idx)

def cleanse_X(X,n_steps, end_islands):

    #1. Expand start_islands and end_islands to obtain an array defining non valid indeces
    X_c = X.copy()

    for ei in end_islands:
        for i in np.arange(n_steps-1):
            if(np.isin(ei+i+1,X_c.index)):
                X_c = X_c.drop(ei+i+1)

    #TODO: check sul fatto che potrei voler droppare roba che fa parte di 
    #un island vicina (succede quando ho un solo campione in mezzo a due islands)

    #TODO: controlla che non succeda la stessa roba sopra

    #Possibile soluzione schifosa: fai ciclo che toglie controllando elemento per elemento
    #usando un np.isin
    return X_c