import numpy as np
from sklearn.svm import SVC
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import StandardScaler
class ReseauNeurones:
    def __init__(self, donnees_train, donnees_eval):
        self.donnees_train = donnees_train
        self.donnees_eval = donnees_eval
        self.svm = None

        np.random.shuffle(self.donnees_train)
        self.nombre_lignes_base = donnees_train.shape[0]
        self.nombre_colonnes_base = donnees_train.shape[1]

        self.x_train = self.donnees_train[0:round(self.nombre_lignes_base * 3 / 4), :self.donnees_train.shape[1] - 1]
        self.y_train = self.donnees_train[0:round(self.nombre_lignes_base * 3 / 4), self.donnees_train.shape[1] - 1:]
        self.y_train = column_or_1d(self.y_train, warn=False)


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
        self.x_test = scaler.transform(self.x_test)

        self.x_to_predict = scaler.transform(self.x_to_predict)

    def predict(self):
        self.svm = SVC(gamma='auto')
        self.svm.fit(self.x_train, self.y_train)
        self.y_predicted = self.svm.predict(self.x_to_predict)
        return self.y_predicted