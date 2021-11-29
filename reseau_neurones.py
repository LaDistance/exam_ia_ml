import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

class ReseauNeurones(donnees_train, donnees_eval, nb_neurones):
    def __init__(self):
        self.donnees_data_frame = donnees_train
        self.donnees_eval = donnees_eval
        self_nb_neurones = nb_neurones

    def train(self):
        nbr_iterations = 20
        i = 0
        metrics_scores = []
        while (i < 50) and (not early_stopping):
            model = MLPClassifier(hidden_layer_sizes=[nbr_neurones], random_state=7, max_iter=nbr_iterations)
            model.fit(x_train, y_train)

            y_predit_train = model.predict(x_train)
            y_predit_validation = model.predict(x_validation)
            y_predit_test = model.predict(x_test)

            print("Taux de reconnaissance en test :")
            print("----------------------")
            metrics_scores.append(metrics.accuracy_score(y_test, y_predit_test) * 100)
            print(metrics.accuracy_score(y_test, y_predit_test) * 100)

            if i == 0:
                tableau_erreurs_train = np.array(100 - metrics.accuracy_score(y_train, y_predit_train) * 100)
                tableau_erreurs_validation = np.array(
                    100 - metrics.accuracy_score(y_validation, y_predit_validation) * 100)
                tableau_erreurs_test = np.array(100 - metrics.accuracy_score(y_test, y_predit_test) * 100)
            else:
                tableau_erreurs_train = np.append(tableau_erreurs_train,
                                                  100 - metrics.accuracy_score(y_train, y_predit_train) * 100)
                tableau_erreurs_validation = np.append(tableau_erreurs_validation,
                                                       100 - metrics.accuracy_score(y_validation,
                                                                                    y_predit_validation) * 100)
                tableau_erreurs_test = np.append(tableau_erreurs_test,
                                                 100 - metrics.accuracy_score(y_test, y_predit_test) * 100)

            if (i > 4):  # on voudrait pas que ça ne stoppe trop vite...
                erreur_validation_courante = tableau_erreurs_validation[tableau_erreurs_validation.size - 1]
                erreur_validation_la_plus_basse = min(tableau_erreurs_validation)

                print("erreur_validation_courante =" + str(erreur_validation_courante))
                print("erreur_validation_la_plus_basse =" + str(min(tableau_erreurs_validation)))

                if (
                        erreur_validation_courante >= 1.10 * erreur_validation_la_plus_basse):  # i>4 pour laisser le réseau s'élancer
                    print(f"Itération d'arrêt : {i} - Erreur actuelle : {erreur_validation_courante}")
                    early_stopping = True;

            i = i + 1
            nbr_iterations = nbr_iterations + 20


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

donnees_data_frame_eval = pd.read_csv ("test_EVALUATION.txt" , delimiter=" ")
donnees_eval_total= donnees_data_frame_eval.values
nombre_lignes_eval = donnees_eval_total.shape[0]
nombre_colonnes_eval = donnees_eval_total.shape[1]

x_to_predict = donnees_eval_total[0:round(nombre_lignes_eval), :nombre_colonnes_eval-1]
y_to_predict = donnees_eval_total[0:round(nombre_lignes_eval), nombre_colonnes_eval-1:]

scaler = StandardScaler (with_mean=True, with_std=True)
scaler.fit (x_train)

x_train= scaler.transform (x_train)
x_validation= scaler.transform (x_validation)
x_test= scaler.transform (x_test)

x_to_predict = scaler.transform(x_to_predict)
nbr_neurones=5
early_stopping = False
erreur_validation_la_plus_basse=100





y_predicted = model.predict(x_to_predict)
print(repr(y_predicted))
