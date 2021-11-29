import pandas as pd
import numpy as np
from sklearn.utils.validation import column_or_1d
from sklearn.svm import SVC
from sklearn import metrics


class SVM():
    donnees_data_frame = pd.read_csv("train.txt", delimiter=" ")
    donnees_ensemble_total = donnees_data_frame.values
    print(donnees_ensemble_total)

    np.random.shuffle(donnees_ensemble_total)

    nombre_lignes_base = donnees_ensemble_total.shape[0]
    nombre_colonnes_base = donnees_ensemble_total.shape[1]

    x_train = donnees_ensemble_total[0:round(nombre_lignes_base * 3 / 4), :donnees_ensemble_total.shape[1] - 1]
    y_train = donnees_ensemble_total[0:round(nombre_lignes_base * 3 / 4), donnees_ensemble_total.shape[1] - 1:]

    x_test = donnees_ensemble_total[round(nombre_lignes_base * 3 / 4) + 1:, :donnees_ensemble_total.shape[1] - 1]
    y_test = donnees_ensemble_total[round(nombre_lignes_base * 3 / 4) + 1:, donnees_ensemble_total.shape[1] - 1:]

    y_train = column_or_1d(y_train, warn=False)
    y_test = column_or_1d(y_test, warn=False)


    svm = SVC(gamma='auto')

    print("\n\n===================\n")

    # apprentissage – construction du modèle prédictif
    svm.fit(x_train, y_train)

    y_predit_test = svm.predict(x_test)
    print(f"y_predit_test {y_predit_test}")

    err = (1.0 - metrics.accuracy_score(y_test, y_predit_test)) * 100
    print("Erreur = ", round(err, 2), "%")
    print("\n\n===================")