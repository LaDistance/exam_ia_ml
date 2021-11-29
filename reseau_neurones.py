import pandas as pd
import numpy as np
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

donnees_data_frame = pd.read_csv ("train.txt" , delimiter=" ")
donnees_ensemble_total= donnees_data_frame.values
print (donnees_ensemble_total)

np.random.shuffle(donnees_ensemble_total)

nombre_lignes_base=donnees_ensemble_total.shape[0]
nombre_colonnes_base=donnees_ensemble_total.shape[1]

x_train = donnees_ensemble_total[0:round(nombre_lignes_base*3/4*2/3),:donnees_ensemble_total.shape[1]-1]
y_train = donnees_ensemble_total[0:round(nombre_lignes_base*3/4*2/3),donnees_ensemble_total.shape[1]-1:]
y_train = column_or_1d (y_train, warn=False)

x_validation = donnees_ensemble_total[round(nombre_lignes_base*3/4*2/3)+1:round(nombre_lignes_base*3/4),:donnees_ensemble_total.shape[1]-1]
y_validation = donnees_ensemble_total[round(nombre_lignes_base*3/4*2/3)+1:round(nombre_lignes_base*3/4),donnees_ensemble_total.shape[1]-1:]
y_validation = column_or_1d (y_validation, warn=False)

x_test = donnees_ensemble_total[round(nombre_lignes_base*3/4)+1:,:donnees_ensemble_total.shape[1]-1]
y_test = donnees_ensemble_total[round(nombre_lignes_base*3/4)+1:,donnees_ensemble_total.shape[1]-1:]
y_test = column_or_1d (y_test, warn=False)


print(f"Nombre de lignes : {nombre_lignes_base} -- nombre de colonnes : {nombre_colonnes_base}")
print(f"Nombre de lignes de x_train : {x_train.shape[0]}")
print(f"Nombre de colonnes de x_train : {x_train.shape[1]}")
print(f"x_train : {x_train} -- y_train : {y_train}")

scaler = StandardScaler (with_mean=True, with_std=True)
scaler.fit (x_train)

x_train= scaler.transform (x_train)
x_validation= scaler.transform (x_validation)
x_test= scaler.transform (x_test)

nbr_neurones=1
early_stopping = False
nbr_iterations=20
i=0
while(i<50) and (not early_stopping):
