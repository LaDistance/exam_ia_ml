import numpy as np
np.random.seed(7)
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

class ArbreDecision:
    def __init__(self, donnees_train, donnees_eval):
        self.donnees_eval = donnees_eval
        self.donnees_train = donnees_train

        np.random.shuffle(self.donnees_train)

        self.nombre_lignes_base=self.donnees_train.shape[0]
        self.nombre_colonnes_base=self.donnees_train.shape[1]

        self.x_train = self.donnees_train[0:round(self.nombre_lignes_base*2/3),:self.donnees_train.shape[1]-1]
        self.y_train = self.donnees_train[0:round(self.nombre_lignes_base*2/3),self.donnees_train.shape[1]-1:]

        self.x_validation = self.donnees_train[round(self.nombre_lignes_base*3/4*2/3)+1:round(self.nombre_lignes_base*3/4),:self.donnees_train.shape[1]-1]
        self.y_validation = self.donnees_train[round(self.nombre_lignes_base*3/4*2/3)+1:round(self.nombre_lignes_base*3/4),self.donnees_train.shape[1]-1:]

        self.x_test = self.donnees_train[round(self.nombre_lignes_base*2/3)+1:,:self.donnees_train.shape[1]-1]
        self.y_test = self.donnees_train[round(self.nombre_lignes_base*2/3)+1:,self.donnees_train.shape[1]-1:]

        self.nombre_lignes_eval = self.donnees_eval.shape[0]
        self.nombre_colonnes_eval = self.donnees_eval.shape[1]

        self.x_to_predict = self.donnees_eval[0:round(self.nombre_lignes_eval), :self.nombre_colonnes_eval - 1]
        self.y_to_predict = self.donnees_eval[0:round(self.nombre_lignes_eval), self.nombre_colonnes_eval - 1:]

        self.y_predit_result = None
        self.y_predit_validation = None
        self.y_predit_train = None

        self.tree = None


    def scale(self):
        scaler = StandardScaler (with_mean=True, with_std=True)
        scaler.fit (self.x_train)
        self.x_train= scaler.transform(self.x_train)
        self.x_test= scaler.transform(self.x_test)

    def train(self):
        early_stopping = False;
        max_depth_courante=0
        metrics_scores = []

        while (max_depth_courante<50) and (not early_stopping) :
            self.tree = DecisionTreeClassifier(max_depth=6)
            self.tree.fit(self.x_train,self.y_train)
            self.y_predit_result = self.tree.predict(self.x_test)
            self.y_predit_validation = self.tree.predict(self.x_validation)
            self.y_predit_train = self.tree.predict(self.x_train)
            metrics_scores.append(metrics.accuracy_score(self.y_test, self.y_predit_result) * 100)
            if max_depth_courante == 0:
                tableau_erreurs_train = np.array(100 - metrics.accuracy_score(self.y_train, self.y_predit_train) * 100)
                tableau_erreurs_validation = np.array(100 - metrics.accuracy_score(self.y_validation, self.y_predit_validation) * 100)
                tableau_erreurs_test = np.array(100 - metrics.accuracy_score(self.y_test, self.y_predit_result) * 100)
            else:
                tableau_erreurs_train = np.append(tableau_erreurs_train,
                                                  100 - metrics.accuracy_score(self.y_train, self.y_predit_train) * 100)
                tableau_erreurs_validation = np.append(tableau_erreurs_validation,
                                                       100 - metrics.accuracy_score(self.y_validation, self.y_predit_validation) * 100)
                tableau_erreurs_test = np.append(tableau_erreurs_test,
                                                 100 - metrics.accuracy_score(self.y_test, self.y_predit_result) * 100)

            if max_depth_courante > 4:  # on voudrait pas que ça ne stoppe trop vite...
                erreur_validation_courante = tableau_erreurs_validation[tableau_erreurs_validation.size - 1]
                erreur_validation_la_plus_basse = min(tableau_erreurs_validation)

                if erreur_validation_courante >= 1.10 * erreur_validation_la_plus_basse:  # i>4 pour laisser le réseau s'élancer
                    early_stopping = True;

            max_depth_courante = max_depth_courante + 1

    def predict(self):
        return self.y_predit_result

if __name__ == '__main__':
    donnees_data_frame = pd.read_csv("train.txt", delimiter=" ")
    donnees_ensemble_total = donnees_data_frame.values

    donnees_data_frame_eval = pd.read_csv("test_EVALUATION.txt", delimiter=" ")
    donnees_eval_total = donnees_data_frame_eval.values
    arbreDecision = ArbreDecision(donnees_ensemble_total, donnees_eval_total)

    arbreDecision.scale()
    arbreDecision.train()
    vecteur_sortie_arbre = arbreDecision.predict()

    print(repr(vecteur_sortie_arbre))