import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from numpy.random import seed
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense , LSTM, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf
from itertools import product


############################################################### Nacitanie a uprava dat######################################################################
sezona1 = pd.read_csv("LaLiga_dataset/LaLiga_2014-2015.csv")
sezona2 = pd.read_csv("LaLiga_dataset/LaLiga_2015-2016.csv")
sezona3 = pd.read_csv("LaLiga_dataset/LaLiga_2016-2017.csv")
sezona4 = pd.read_csv("LaLiga_dataset/LaLiga_2017-2018.csv")
sezona5 = pd.read_csv("LaLiga_dataset/LaLiga_2018-2019.csv")
sezona6 = pd.read_csv("LaLiga_dataset/LaLiga_2019-2020.csv")
sezona7 = pd.read_csv("LaLiga_dataset/LaLiga_2020-2021.csv")
sezona8 = pd.read_csv("LaLiga_dataset/LaLiga_2021-2022.csv")

# Spojenie vsetkych sezón do jednej tabulky
vsetky_sezony = pd.concat([sezona1, sezona2, sezona3, sezona4, sezona5, sezona6, sezona7, sezona8], ignore_index=True)
selected_columns = ['HomeTeam', 'AwayTeam', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY']
vsetky_sezony = vsetky_sezony[selected_columns]

X = vsetky_sezony.drop('FTR', axis=1) 
y = vsetky_sezony['FTR'] 

print(f"V celej datovej sade LaLiga sa nachadza: Vyhry: {sum(y == 'H')}, Prehry: {sum(y == 'A')}, Remizy: {sum(y == 'D')}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Rozdelenie dat na trenovaciu a testovaciu mnozinu
print(f"V testovacej sade LaLiga sa nachadza: Vyhry: {sum(y_test == 'H')}, Prehry: {sum(y_test == 'A')}, Remizy: {sum(y_test == 'D')}")

# Pouzivam LabelEncoder na zakodovanie vysledkov do ciselnej podoby
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Pouzivam ColumnTransformer na zakodovanie kategorickych atributov a normalizaciu ciselnych atributov
ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['HomeTeam', 'AwayTeam']),
        ('scaler', StandardScaler(), ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY'])
    ],
    remainder='passthrough'  
)

X_train_encoded = ct.fit_transform(X_train)
X_test_encoded = ct.transform(X_test)



############################################################### Jednovrstvovy perceptron KERAS ######################################################################

def single_layer_perceptron_keras(X_train_encoded, y_train, X_test_encoded, y_test):
    def create_model(optimizer='adam', activation='relu'):
        model = Sequential()
        model.add(Dense(3, input_dim=X_train_encoded.shape[1], activation=activation))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    model = KerasClassifier(model=create_model, verbose=0)
    param_grid = {
        'model__optimizer': ['adam', 'Adadelta', 'SGD', 'rmsprop'],
        'model__activation': ['softmax', 'sigmoid', 'relu', 'tanh'],
        'epochs': [15],
        'batch_size': [32],
        'random_state': [42],
    }

    # Vsetky kombinacie parametrov
    all_param_combinations = list(product(*param_grid.values()))
    print()
    print('---------------------------------------------[ Jednovrstvovy perceptron KERAS ]---------------------------------------------')
    print('Pocet kombinacii: ', len(all_param_combinations))
    print('--------')
    results = []
    for params in all_param_combinations:     # Robim modely pre vsetky kombinacie parametrov
        model.set_params(**dict(zip(param_grid.keys(), params))) 
        model.fit(X_train_encoded, y_train)
        y_pred = model.predict(X_test_encoded)

        accuracy = (y_pred == y_test).mean()
        predikovane_vyhry = sum(y_pred == 2)
        predikovane_prehry = sum(y_pred == 0)
        predikovane_remizy = sum(y_pred == 1)

        spravne_vyhry = sum((y_pred == y_test) & (y_test == 2))
        spravne_prehry = sum((y_pred == y_test) & (y_test == 0))
        spravne_remizy = sum((y_pred == y_test) & (y_test == 1))
        results.append((params, accuracy, predikovane_vyhry, predikovane_prehry, predikovane_remizy, spravne_vyhry, spravne_prehry, spravne_remizy))

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True) # Zoradujem modely podla presnosti
    # Vypisujem modely
    for i, (params, accuracy, predikovane_vyhry, predikovane_prehry, predikovane_remizy, spravne_vyhry, spravne_prehry, spravne_remizy) in enumerate(sorted_results[:16], 1):
        optimizer = params[0]
        activation = params[1]
        print(f"Model {i} parametre: aktivacna funkcia = {activation}, optimizer = {optimizer}")
        print(f"Presnost modelu {i}: {accuracy*100:.3f}%")
        print(f"Predikovane vysledky:               Vyhry: {predikovane_vyhry}, Prehry: {predikovane_prehry}, Remizy: {predikovane_remizy}")
        print(f"Predikovane spravne vysledky:       Vyhry: {spravne_vyhry}, Prehry: {spravne_prehry}, Remizy: {spravne_remizy}")
        print('--------')




############################################################### Jednovrstvovy perceptron MLPC Scikit ######################################################################
def single_layer_perceptron_mlpc(X_train_encoded, y_train, X_test_encoded, y_test):
    # Definícia mriežky parametrov
    param_grid = {
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'hidden_layer_sizes': [10, 20, 30 , 50, 100],
        'max_iter': [10, 50, 100],
        'batch_size': [32],
        'random_state': [42],
    }
    model = MLPClassifier()

    # Vsetky kombinacie parametrov
    all_param_combinations = list(product(*param_grid.values()))
    print()
    print('---------------------------------------------[ Jednovrstvovy perceptron MLPC Scikit ]---------------------------------------------')
    print('Pocet kombinacii: ', len(all_param_combinations))
    print('Najlepsie 3 modely: ')
    print('--------')
    results = []
    for params in all_param_combinations:     # Robim modely pre vsetky kombinacie parametrov
        model.set_params(**dict(zip(param_grid.keys(), params))) 
        model.fit(X_train_encoded, y_train)
        y_pred = model.predict(X_test_encoded)

        accuracy = (y_pred == y_test).mean()
        predikovane_vyhry = sum(y_pred == 2)
        predikovane_prehry = sum(y_pred == 0)
        predikovane_remizy = sum(y_pred == 1)

        spravne_vyhry = sum((y_pred == y_test) & (y_test == 2))
        spravne_prehry = sum((y_pred == y_test) & (y_test == 0))
        spravne_remizy = sum((y_pred == y_test) & (y_test == 1))
        results.append((params, accuracy, predikovane_vyhry, predikovane_prehry, predikovane_remizy, spravne_vyhry, spravne_prehry, spravne_remizy))

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True) # Zoradujem modely podla presnosti
    # Vypisujem modely
    for i, (params, accuracy, predikovane_vyhry, predikovane_prehry, predikovane_remizy, spravne_vyhry, spravne_prehry, spravne_remizy) in enumerate(sorted_results[:3], 1):
        activation = params[0]
        solver = params[1]
        layers = params[2]
        max_iter = params[3]
        print(f"Model {i} parametre: aktivacna funkcia = {activation} solver = {solver}, hidden_layer_sizes = {layers}, max_iter = {max_iter}")
        print(f"Presnost modelu {i}: {accuracy*100:.3f}%")
        print(f"Predikovane vysledky:               Vyhry: {predikovane_vyhry}, Prehry: {predikovane_prehry}, Remizy: {predikovane_remizy}")
        print(f"Predikovane spravne vysledky:       Vyhry: {spravne_vyhry}, Prehry: {spravne_prehry}, Remizy: {spravne_remizy}")
        print('--------')
 



############################################################### Viacvrstvovy perceptron KERAS ######################################################################
def mlp(X_train_encoded, y_train, X_test_encoded, y_test):

    def create_model(optimizer='adam', activation1='relu', activation2='relu', activation3='relu'):
        model = Sequential()
        model.add(Dense(74, input_dim= X_train_encoded.shape[1], activation=activation1))
        model.add(Dense(32, activation=activation2))
        model.add(Dense(3, activation=activation3))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    model = KerasClassifier(build_fn=create_model, verbose=0)
    param_grid = {
        'model__activation1': ['relu', 'tanh', 'sigmoid', 'softmax'],
        'model__activation2': ['relu', 'tanh','sigmoid', 'softmax'],
        'model__activation3': ['relu', 'tanh', 'sigmoid', 'softmax'],
        'model__optimizer': ['adam', 'sgd', 'rmsprop', 'Adadelta'],
        'epochs': [10],
        'batch_size': [32],
        'random_state': [42],
    }

    # Vsetky kombinacie parametrov
    all_param_combinations = list(product(*param_grid.values()))
    print()
    print('---------------------------------------------[ Viacvrstvovy perceptron KERAS ]---------------------------------------------')
    print('Pocet kombinacii: ', len(all_param_combinations))
    print('Najlepsie 3 modely: ')
    print('--------')
    results = []
    for params in all_param_combinations:     # Robim modely pre vsetky kombinacie parametrov
        model.set_params(**dict(zip(param_grid.keys(), params))) 
        model.fit(X_train_encoded, y_train)
        y_pred = model.predict(X_test_encoded)

        accuracy = (y_pred == y_test).mean()
        predikovane_vyhry = sum(y_pred == 2)
        predikovane_prehry = sum(y_pred == 0)
        predikovane_remizy = sum(y_pred == 1)

        spravne_vyhry = sum((y_pred == y_test) & (y_test == 2))
        spravne_prehry = sum((y_pred == y_test) & (y_test == 0))
        spravne_remizy = sum((y_pred == y_test) & (y_test == 1))
        results.append((params, accuracy, predikovane_vyhry, predikovane_prehry, predikovane_remizy, spravne_vyhry, spravne_prehry, spravne_remizy))

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True) # Zoradujem modely podla presnosti
    # Vypisujem modely
    for i, (params, accuracy, predikovane_vyhry, predikovane_prehry, predikovane_remizy, spravne_vyhry, spravne_prehry, spravne_remizy) in enumerate(sorted_results[:3], 1):
        activation1 = params[0]
        activation2 = params[1]
        activation3 = params[2]
        optimizer = params[3]
        print(f"Model {i} parametre: aktivacna funkcia 1 = {activation1}, aktivacna funkcia 2 = {activation2}, aktivacna funkcia 3 = {activation3}, optimizer = {optimizer}")
        print(f"Presnost modelu {i}: {accuracy*100:.3f}%")
        print(f"Predikovane vysledky:               Vyhry: {predikovane_vyhry}, Prehry: {predikovane_prehry}, Remizy: {predikovane_remizy}")
        print(f"Predikovane spravne vysledky:       Vyhry: {spravne_vyhry}, Prehry: {spravne_prehry}, Remizy: {spravne_remizy}")
        print('--------')




############################################################### Viacvrstvovy perceptron MLPC Scikit ######################################################################
def mlpc(X_train_encoded, y_train, X_test_encoded, y_test):
    # Definícia mriežky parametrov
    param_grid = {
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd', 'lbfgs'],
    'hidden_layer_sizes': [
        (100, 50, 3),
        (80, 40, 3),
        (120, 60, 3),
        (150, 75, 3),
        (90, 45, 3),
        (110, 55, 3),
        (130, 65, 3),
        (140, 70, 3),
    ],
    'max_iter': [50, 100],
    'batch_size': [32],
    'random_state': [42],
}
    model = MLPClassifier()

    # Vsetky kombinacie parametrov
    all_param_combinations = list(product(*param_grid.values()))
    print()
    print('---------------------------------------------[ Viacvrstvovy perceptron MLPC Scikit ]---------------------------------------------')
    print('Pocet kombinacii: ', len(all_param_combinations))
    print('Najlepsie 3 modely: ')
    print('--------')
    results = []
    for params in all_param_combinations:     # Robim modely pre vsetky kombinacie parametrov
        model.set_params(**dict(zip(param_grid.keys(), params))) 
        model.fit(X_train_encoded, y_train)
        y_pred = model.predict(X_test_encoded)

        accuracy = (y_pred == y_test).mean()
        predikovane_vyhry = sum(y_pred == 2)
        predikovane_prehry = sum(y_pred == 0)
        predikovane_remizy = sum(y_pred == 1)

        spravne_vyhry = sum((y_pred == y_test) & (y_test == 2))
        spravne_prehry = sum((y_pred == y_test) & (y_test == 0))
        spravne_remizy = sum((y_pred == y_test) & (y_test == 1))
        results.append((params, accuracy, predikovane_vyhry, predikovane_prehry, predikovane_remizy, spravne_vyhry, spravne_prehry, spravne_remizy))

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True) # Zoradujem modely podla presnosti
    # Vypisujem modely
    for i, (params, accuracy, predikovane_vyhry, predikovane_prehry, predikovane_remizy, spravne_vyhry, spravne_prehry, spravne_remizy) in enumerate(sorted_results[:3], 1):
        activation = params[0]
        solver = params[1]
        layers = params[2]
        max_iter = params[3]
        print(f"Model {i} parametre: aktivacna funkcia = {activation} solver = {solver}, hidden_layer_sizes = {layers}, max_iter = {max_iter}")
        print(f"Presnost modelu {i}: {accuracy*100:.3f}%")
        print(f"Predikovane vysledky:               Vyhry: {predikovane_vyhry}, Prehry: {predikovane_prehry}, Remizy: {predikovane_remizy}")
        print(f"Predikovane spravne vysledky:       Vyhry: {spravne_vyhry}, Prehry: {spravne_prehry}, Remizy: {spravne_remizy}")
        print('--------')
      



############################################################### LSTM KERAS ######################################################################
def lstm_in_keras(X_train_encoded, y_train, X_test_encoded, y_test):

    def create_model(optimizer='adam', dropout_rate=0.2, activation='relu'):    

        model = Sequential()
        model.add(LSTM(74, input_shape=(X_train_encoded.shape[1], 1), return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(3, activation=activation))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    # Convert sparse matrices to dense arrays
    X_train_dense = X_train_encoded.toarray()
    X_test_dense = X_test_encoded.toarray()

    # Vytvorenie modelu
    model = KerasClassifier(build_fn=create_model, verbose=0)

    # Definícia mriežky parametrov
    param_grid = {
        'model__activation': ['sigmoid', 'softmax', 'tanh', 'relu'],
        'model__optimizer': ['adam', 'sgd', 'rmsprop'],
        'model__dropout_rate': [0.2],
        'epochs': [8,10,15],
        'batch_size': [32],
        'random_state': [42],   }

    # Vsetky kombinacie parametrov
    all_param_combinations = list(product(*param_grid.values()))
    print()
    print()
    print('---------------------------------------------[ LSTM KERAS ]---------------------------------------------')
    print('Pocet kombinacii: ', len(all_param_combinations))
    print('Najlepsie 3 modely: ')
    print('--------')
    results = []
    for params in all_param_combinations:    # Robim modely pre vsetky kombinacie parametrov
        model.set_params(**dict(zip(param_grid.keys(), params))) 
        model.fit(X_train_dense, y_train)
        y_pred = model.predict(X_test_dense)

        accuracy = (y_pred == y_test).mean()
        predikovane_vyhry = sum(y_pred == 2)
        predikovane_prehry = sum(y_pred == 0)
        predikovane_remizy = sum(y_pred == 1)

        spravne_vyhry = sum((y_pred == y_test) & (y_test == 2))
        spravne_prehry = sum((y_pred == y_test) & (y_test == 0))
        spravne_remizy = sum((y_pred == y_test) & (y_test == 1))
        results.append((params, accuracy, predikovane_vyhry, predikovane_prehry, predikovane_remizy, spravne_vyhry, spravne_prehry, spravne_remizy))
    # Zoradujem modely podla presnosti
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    # Vypisujem modely
    for i, (params, accuracy, predikovane_vyhry, predikovane_prehry, predikovane_remizy, spravne_vyhry, spravne_prehry, spravne_remizy) in enumerate(sorted_results[:3], 1):
        activation = params[0]
        optimizer = params[1]
        dropout = params[2]
        epochs = params[3]
        print(f"Model {i} parametre: aktivacna funkcia = {activation}, optimizer = {optimizer}, dropout_rate = {dropout}, epochs = {epochs}")
        print(f"Presnost modelu {i}: {accuracy*100:.3f}%")
        print(f"Predikovane vysledky:               Vyhry: {predikovane_vyhry}, Prehry: {predikovane_prehry}, Remizy: {predikovane_remizy}")
        print(f"Predikovane spravne vysledky:       Vyhry: {spravne_vyhry}, Prehry: {spravne_prehry}, Remizy: {spravne_remizy}")
        print('--------')


single_layer_perceptron_keras(X_train_encoded, y_train, X_test_encoded, y_test)
single_layer_perceptron_mlpc(X_train_encoded, y_train, X_test_encoded, y_test)   
mlp(X_train_encoded, y_train, X_test_encoded, y_test)
mlpc(X_train_encoded, y_train, X_test_encoded, y_test)
lstm_in_keras(X_train_encoded, y_train, X_test_encoded, y_test)  