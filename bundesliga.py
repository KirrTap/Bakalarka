import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense , LSTM, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from itertools import product
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
############################################################### Nacitanie a uprava dat######################################################################
sezona1 = pd.read_csv("BundesLiga_dataset/BundesLiga_2014-2015.csv")
sezona2 = pd.read_csv("BundesLiga_dataset/BundesLiga_2015-2016.csv")
sezona3 = pd.read_csv("BundesLiga_dataset/BundesLiga_2016-2017.csv")
sezona4 = pd.read_csv("BundesLiga_dataset/BundesLiga_2017-2018.csv")
sezona5 = pd.read_csv("BundesLiga_dataset/BundesLiga_2018-2019.csv")
sezona6 = pd.read_csv("BundesLiga_dataset/BundesLiga_2019-2020.csv")
sezona7 = pd.read_csv("BundesLiga_dataset/BundesLiga_2020-2021.csv")
sezona8 = pd.read_csv("BundesLiga_dataset/BundesLiga_2021-2022.csv")


# Spojenie vsetkych sezón do jednej tabulky
vsetky_sezony = pd.concat([sezona1, sezona2, sezona3, sezona4, sezona5, sezona6, sezona7, sezona8], ignore_index=True)
selected_columns = ['HomeTeam', 'AwayTeam', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY']
vsetky_sezony = vsetky_sezony[selected_columns]

X = vsetky_sezony.drop('FTR', axis=1) 
y = vsetky_sezony['FTR'] 

print(f"V celej datovej sade BundesLiga sa nachadza: Vyhry: {sum(y == 'H')}, Prehry: {sum(y == 'A')}, Remizy: {sum(y == 'D')}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Rozdelenie dat na trenovaciu a testovaciu mnozinu
print(f"V testovacej sade BundesLiga sa nachadza: Vyhry: {sum(y_test == 'H')}, Prehry: {sum(y_test == 'A')}, Remizy: {sum(y_test == 'D')}")

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

   
    param_grid = {
        'model__optimizer': ['adam', 'Adadelta', 'SGD', 'rmsprop'],
        'model__activation': ['softmax', 'sigmoid', 'relu', 'tanh'],
        'epochs': [15],
        'batch_size': [32]
    }

    model = KerasClassifier(build_fn=create_model, verbose=0)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_encoded, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_encoded)
    accuracy = accuracy_score(y_test, y_pred)

    cv_scores = cross_val_score(best_model, X_train_encoded, y_train, cv=5, scoring='accuracy')

    print()
    print('---------------------------------------------[ Jednovrstvovy perceptron KERAS ]---------------------------------------------')
    print("Pocet kombinacii: ", len(list(product(*param_grid.values()))))
    print('Najlepsie parametre modelu: ', grid_search.best_params_)
    print("Cross-validacia skore: ", cv_scores)
    print(f"Priemerna presnost modelu pri krizovej validacii: {cv_scores.mean()*100:.3f}%")
    print(f"Rozpyl presnosti modelu pri krizovej validacii: ± {cv_scores.std()*100:.3f}%")
    print(f"Presnost modelu na testovacej sade: {accuracy*100:.3f}%")
    print('--------')   
    print(f"Predikovane vysledky:               Vyhry: {sum(y_pred == 2)}, Prehry: {sum(y_pred == 0)}, Remizy: {sum(y_pred == 1)}")
    print(f"Predikovane spravne vysledky:       Vyhry: {sum((y_pred == y_test) & (y_test == 2))}, Prehry: {sum((y_pred == y_test) & (y_test == 0))}, Remizy: {sum((y_pred == y_test) & (y_test == 1))}")




############################################################### Jednovrstvovy perceptron MLPC Scikit ######################################################################
def single_layer_perceptron_mlpc(X_train_encoded, y_train, X_test_encoded, y_test):

    param_grid = {
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'hidden_layer_sizes': [10, 20, 30 , 50, 100],
        'max_iter': [10, 50, 100],
        'batch_size': [32],
    }
    model = MLPClassifier()

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_encoded, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_encoded)
    accuracy = accuracy_score(y_test, y_pred)

    cv_scores = cross_val_score(best_model, X_train_encoded, y_train, cv=5, scoring='accuracy')

    print()
    print('---------------------------------------------[ Jednovrstvovy perceptron MLPC Scikit ]---------------------------------------------')
    print("Pocet kombinacii: ", len(list(product(*param_grid.values()))))
    print('Najlepsie parametre modelu: ', grid_search.best_params_)
    print("Cross-validacia skore: ", cv_scores)
    print(f"Priemerna presnost modelu pri krizovej validacii: {cv_scores.mean()*100:.3f}%")
    print(f"Rozpyl presnosti modelu pri krizovej validacii: ± {cv_scores.std()*100:.3f}%")
    print(f"Presnost modelu na testovacej sade: {accuracy*100:.3f}%")
    print('--------')
    print(f"Predikovane vysledky:               Vyhry: {sum(y_pred == 2)}, Prehry: {sum(y_pred == 0)}, Remizy: {sum(y_pred == 1)}")
    print(f"Predikovane spravne vysledky:       Vyhry: {sum((y_pred == y_test) & (y_test == 2))}, Prehry: {sum((y_pred == y_test) & (y_test == 0))}, Remizy: {sum((y_pred == y_test) & (y_test == 1))}")    



############################################################### Viacvrstvovy perceptron KERAS ######################################################################
def mlp(X_train_encoded, y_train, X_test_encoded, y_test):

    def create_model(optimizer='adam', activation1='relu', activation2='relu', activation3='relu'):
        model = Sequential()
        model.add(Dense(74, input_dim= X_train_encoded.shape[1], activation=activation1))
        model.add(Dense(32, activation=activation2))
        model.add(Dense(3, activation=activation3))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    
    param_grid = {
        'model__activation1': ['relu', 'tanh', 'sigmoid', 'softmax'],
        'model__activation2': ['relu', 'tanh','sigmoid', 'softmax'],
        'model__activation3': ['relu', 'tanh', 'sigmoid', 'softmax'],
        'model__optimizer': ['adam', 'sgd', 'rmsprop', 'Adadelta'],
        'epochs': [10],
        'batch_size': [32],
    }

    model = KerasClassifier(build_fn=create_model, verbose=0)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_encoded, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_encoded)
    accuracy = accuracy_score(y_test, y_pred)

    cv_scores = cross_val_score(best_model, X_train_encoded, y_train, cv=5, scoring='accuracy')

    print()
    print('---------------------------------------------[ Viacvrstvovy perceptron KERAS ]---------------------------------------------')
    print("Pocet kombinacii: ", len(list(product(*param_grid.values()))))
    print('Najlepsie parametre modelu: ', grid_search.best_params_)
    print("Cross-validacia skore: ", cv_scores)
    print(f"Priemerna presnost modelu pri krizovej validacii: {cv_scores.mean()*100:.3f}%")
    print(f"Rozpyl presnosti modelu pri krizovej validacii: ± {cv_scores.std()*100:.3f}%")
    print(f"Presnost modelu na testovacej sade: {accuracy*100:.3f}%")
    print('--------')
    print(f"Predikovane vysledky:               Vyhry: {sum(y_pred == 2)}, Prehry: {sum(y_pred == 0)}, Remizy: {sum(y_pred == 1)}")
    print(f"Predikovane spravne vysledky:       Vyhry: {sum((y_pred == y_test) & (y_test == 2))}, Prehry: {sum((y_pred == y_test) & (y_test == 0))}, Remizy: {sum((y_pred == y_test) & (y_test == 1))}")

   




############################################################### Viacvrstvovy perceptron MLPC Scikit ######################################################################
def mlpc(X_train_encoded, y_train, X_test_encoded, y_test):

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

}
    model = MLPClassifier()

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_encoded, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_encoded)
    accuracy = accuracy_score(y_test, y_pred)

    cv_scores = cross_val_score(best_model, X_train_encoded, y_train, cv=5, scoring='accuracy')

    print()
    print('---------------------------------------------[ Viacvrstvovy perceptron MLPC Scikit ]---------------------------------------------')
    print("Pocet kombinacii: ", len(list(product(*param_grid.values()))))
    print('Najlepsie parametre modelu: ', grid_search.best_params_)
    print("Cross-validacia skore: ", cv_scores)
    print(f"Priemerna presnost modelu pri krizovej validacii: {cv_scores.mean()*100:.3f}%")
    print(f"Rozpyl presnosti modelu pri krizovej validacii: ± {cv_scores.std()*100:.3f}%")
    print(f"Presnost modelu na testovacej sade: {accuracy*100:.3f}%")
    print('--------')
    print(f"Predikovane vysledky:               Vyhry: {sum(y_pred == 2)}, Prehry: {sum(y_pred == 0)}, Remizy: {sum(y_pred == 1)}")
    print(f"Predikovane spravne vysledky:       Vyhry: {sum((y_pred == y_test) & (y_test == 2))}, Prehry: {sum((y_pred == y_test) & (y_test == 0))}, Remizy: {sum((y_pred == y_test) & (y_test == 1))}")




############################################################### LSTM KERAS ######################################################################
def lstm_in_keras(X_train_encoded, y_train, X_test_encoded, y_test):

    def create_model(optimizer='adam', dropout_rate=0.2, activation='relu'):    

        model = Sequential()
        model.add(LSTM(74, input_shape=(X_train_encoded.shape[1], 1), return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(3, activation=activation))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    X_train_dense = X_train_encoded.toarray()
    X_test_dense = X_test_encoded.toarray()

    param_grid = {
        'model__activation': ['sigmoid', 'softmax', 'tanh', 'relu'],
        'model__optimizer': ['adam', 'sgd', 'rmsprop'],
        'model__dropout_rate': [0.2],
        'epochs': [8,10,15],
        'batch_size': [32],
  }
    
    model = KerasClassifier(build_fn=create_model, verbose=0)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train_dense, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_dense)
    accuracy = accuracy_score(y_test, y_pred)

    cv_scores = cross_val_score(best_model, X_train_dense, y_train, cv=5, scoring='accuracy')

    print()
    print('---------------------------------------------[ LSTM KERAS ]---------------------------------------------')
    print("Pocet kombinacii: ", len(list(product(*param_grid.values()))))
    print('Najlepsie parametre modelu: ', grid_search.best_params_)
    print("Cross-validacia skore: ", cv_scores)
    print(f"Priemerna presnost modelu pri krizovej validacii: {cv_scores.mean()*100:.3f}%")
    print(f"Rozpyl presnosti modelu pri krizovej validacii: ± {cv_scores.std()*100:.3f}%")
    print(f"Presnost modelu na testovacej sade: {accuracy*100:.3f}%")
    print('--------')
    print(f"Predikovane vysledky:               Vyhry: {sum(y_pred == 2)}, Prehry: {sum(y_pred == 0)}, Remizy: {sum(y_pred == 1)}")
    print(f"Predikovane spravne vysledky:       Vyhry: {sum((y_pred == y_test) & (y_test == 2))}, Prehry: {sum((y_pred == y_test) & (y_test == 0))}, Remizy: {sum((y_pred == y_test) & (y_test == 1))}")




single_layer_perceptron_keras(X_train_encoded, y_train, X_test_encoded, y_test)
single_layer_perceptron_mlpc(X_train_encoded, y_train, X_test_encoded, y_test)   
mlp(X_train_encoded, y_train, X_test_encoded, y_test)
mlpc(X_train_encoded, y_train, X_test_encoded, y_test)
lstm_in_keras(X_train_encoded, y_train, X_test_encoded, y_test)