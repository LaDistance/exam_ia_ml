import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

class ReseauNeurones:
    def __init__(self, donnees_train, donnees_eval, nb_neurones):
        self.donnees_train = donnees_train
        self.donnees_eval = donnees_eval
        self.nb_neurones = nb_neurones
        self.model = None

        np.random.shuffle(self.donnees_train)
        self.nombre_lignes_base = donnees_train.shape[0]
        self.nombre_colonnes_base = donnees_train.shape[1]

        self.x_train = self.donnees_train[0:round(self.nombre_lignes_base * 3 / 4 * 2 / 3),
                  :self.donnees_train.shape[1] - 1]
        self.y_train = self.donnees_train[0:round(self.nombre_lignes_base * 3 / 4 * 2 / 3),
                  self.donnees_train.shape[1] - 1:]
        self.y_train = column_or_1d(self.y_train, warn=False)

        self.x_validation = self.donnees_train[
                       round(self.nombre_lignes_base * 3 / 4 * 2 / 3) + 1:round(self.nombre_lignes_base * 3 / 4),
                       :self.donnees_train.shape[1] - 1]
        self.y_validation = self.donnees_train[
                       round(self.nombre_lignes_base * 3 / 4 * 2 / 3) + 1:round(self.nombre_lignes_base * 3 / 4),
                       self.donnees_train.shape[1] - 1:]
        self.y_validation = column_or_1d(self.y_validation, warn=False)

        self.x_test = self.donnees_train[round(self.nombre_lignes_base * 3 / 4) + 1:, :self.donnees_train.shape[1] - 1]
        self.y_test = self.donnees_train[round(self.nombre_lignes_base * 3 / 4) + 1:, self.donnees_train.shape[1] - 1:]
        self.y_test = column_or_1d(self.y_test, warn=False)

        self.nombre_lignes_eval = self.donnees_eval.shape[0]
        self.nombre_colonnes_eval = self.donnees_eval.shape[1]

        self.x_to_predict = self.donnees_eval[0:round(self.nombre_lignes_eval), :self.nombre_colonnes_eval - 1]
        self.y_to_predict = self.donnees_eval[0:round(self.nombre_lignes_eval), self.nombre_colonnes_eval - 1:]

    def scale(self):
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(self.x_train)

        self.x_train = scaler.transform(self.x_train)
        self.x_validation = scaler.transform(self.x_validation)
        self.x_test = scaler.transform(self.x_test)

        self.x_to_predict = scaler.transform(self.x_to_predict)


    def train(self):
        nbr_iterations = 20
        i = 0
        metrics_scores = []
        erreur_validation_la_plus_basse = 100
        early_stopping = False
        while (i < 50) and (not early_stopping):
            self.model = MLPClassifier(hidden_layer_sizes=[self.nb_neurones], random_state=7, max_iter=nbr_iterations)
            self.model.fit(self.x_train, self.y_train)

            y_predit_train = self.model.predict(self.x_train)
            y_predit_validation = self.model.predict(self.x_validation)
            y_predit_test = self.model.predict(self.x_test)

            print("Taux de reconnaissance en test :")
            print("----------------------")
            metrics_scores.append(metrics.accuracy_score(self.y_test, y_predit_test) * 100)
            print(metrics.accuracy_score(self.y_test, y_predit_test) * 100)

            if i == 0:
                tableau_erreurs_train = np.array(100 - metrics.accuracy_score(self.y_train, y_predit_train) * 100)
                tableau_erreurs_validation = np.array(
                    100 - metrics.accuracy_score(self.y_validation, y_predit_validation) * 100)
                tableau_erreurs_test = np.array(100 - metrics.accuracy_score(self.y_test, y_predit_test) * 100)
            else:
                tableau_erreurs_train = np.append(tableau_erreurs_train,
                                                  100 - metrics.accuracy_score(self.y_train, y_predit_train) * 100)
                tableau_erreurs_validation = np.append(tableau_erreurs_validation,
                                                       100 - metrics.accuracy_score(self.y_validation,
                                                                                    y_predit_validation) * 100)
                tableau_erreurs_test = np.append(tableau_erreurs_test,
                                                 100 - metrics.accuracy_score(self.y_test, y_predit_test) * 100)

            if (i > 4):  # on voudrait pas que ça ne stoppe trop vite...
                erreur_validation_courante = tableau_erreurs_validation[tableau_erreurs_validation.size - 1]
                erreur_validation_la_plus_basse = min(tableau_erreurs_validation)

                print("erreur_validation_courante =" + str(erreur_validation_courante))
                print("erreur_validation_la_plus_basse =" + str(min(tableau_erreurs_validation)))

                if (erreur_validation_courante >= 1.10 * erreur_validation_la_plus_basse):  # i>4 pour laisser le réseau s'élancer
                    print(f"Itération d'arrêt : {i} - Erreur actuelle : {erreur_validation_courante}")
                    early_stopping = True;

            i = i + 1
            nbr_iterations = nbr_iterations + 20
    def predict(self):
        self.y_predicted = self.model.predict(self.x_to_predict)
        return self.y_predicted
