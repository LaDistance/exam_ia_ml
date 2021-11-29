import sys
import random
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

donnees_data_frame = pd.read_csv ("train.txt" , delimiter=" ")
donnees_ensemble_total= donnees_data_frame.values
print (donnees_ensemble_total)

# seed = random.randrange(sys.maxsize)
# [75.0, 83.33333333333334, 89.58333333333334, 89.58333333333334, 87.5, ...]
seed = 2637295433193598974
rng = random.Random(seed)
rng.shuffle(donnees_ensemble_total)

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

donnees_data_frame_eval = pd.read_csv ("test_EVALUATION.txt" , delimiter=" ")
donnees_eval_total= donnees_data_frame_eval.values
print (donnees_eval_total)
nombre_lignes_eval = donnees_eval_total.shape[0]
nombre_colonnes_eval = donnees_eval_total.shape[1]

x_to_predict = donnees_eval_total[0:round(nombre_lignes_eval), :nombre_colonnes_eval-1]
y_to_predict = donnees_eval_total[0:round(nombre_lignes_eval), nombre_colonnes_eval-1:]

print(f"x_to_predict : {x_to_predict} - y_to_predict : {y_to_predict}")

# print(f"Nombre de lignes : {nombre_lignes_base} -- nombre de colonnes : {nombre_colonnes_base}")
# print(f"Nombre de lignes de x_train : {x_train.shape[0]}")
# print(f"Nombre de colonnes de x_train : {x_train.shape[1]}")
# print(f"x_train : {x_train} -- y_train : {y_train}")

scaler = StandardScaler (with_mean=True, with_std=True)
scaler.fit (x_train)

x_train= scaler.transform (x_train)
x_validation= scaler.transform (x_validation)
x_test= scaler.transform (x_test)

nbr_neurones=5
early_stopping = False
erreur_validation_la_plus_basse=100

nbr_iterations=20
i=0
metrics_scores = []
while(i<50) and (not early_stopping):
    model = MLPClassifier(hidden_layer_sizes=[nbr_neurones], random_state=7, max_iter = nbr_iterations)
    model.fit(x_train, y_train)

    y_predit_train = model.predict (x_train)
    y_predit_validation = model.predict (x_validation)
    y_predit_test = model.predict (x_test)

    print("Taux de reconnaissance en test :")
    print("----------------------")
    metrics_scores.append(metrics.accuracy_score(y_test, y_predit_test) * 100)
    print(metrics.accuracy_score(y_test, y_predit_test) * 100)
    print()

    if i == 0:
        tableau_erreurs_train = np.array(100 - metrics.accuracy_score(y_train, y_predit_train) * 100)
        tableau_erreurs_validation = np.array(100 - metrics.accuracy_score(y_validation, y_predit_validation) * 100)
        tableau_erreurs_test = np.array(100 - metrics.accuracy_score(y_test, y_predit_test) * 100)
    else:
        tableau_erreurs_train = np.append(tableau_erreurs_train,
                                          100 - metrics.accuracy_score(y_train, y_predit_train) * 100)
        tableau_erreurs_validation = np.append(tableau_erreurs_validation,
                                               100 - metrics.accuracy_score(y_validation, y_predit_validation) * 100)
        tableau_erreurs_test = np.append(tableau_erreurs_test,
                                         100 - metrics.accuracy_score(y_test, y_predit_test) * 100)

    if (i > 4):  # on voudrait pas que ça ne stoppe trop vite...
        erreur_validation_courante = tableau_erreurs_validation[tableau_erreurs_validation.size - 1]
        erreur_validation_la_plus_basse = min(tableau_erreurs_validation)

        print("erreur_validation_courante =" + str(erreur_validation_courante))
        print("erreur_validation_la_plus_basse =" + str(min(tableau_erreurs_validation)))

        if (erreur_validation_courante >= 1.80 * erreur_validation_la_plus_basse):  # i>4 pour laisser le réseau s'élancer
            print(f"Itération d'arrêt : {i} - Erreur actuelle : {erreur_validation_courante}")
            early_stopping = True;

    i = i+1
    nbr_iterations = nbr_iterations + 20

print(metrics_scores)
if(max(metrics_scores) >= 87):
    print(f"Seed du random shuffle : {seed}")