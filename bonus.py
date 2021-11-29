from reseau_neurones import ReseauNeurones
import pandas as pd

# Partie réseau neurones :
donnees_data_frame = pd.read_csv ("train.txt" , delimiter=" ")
donnees_ensemble_total= donnees_data_frame.values

donnees_data_frame_eval = pd.read_csv ("test_EVALUATION.txt" , delimiter=" ")
donnees_eval_total= donnees_data_frame_eval.values


reseauNeurones = ReseauNeurones(donnees_ensemble_total, donnees_eval_total, nb_neurones=5)

reseauNeurones.scale()
reseauNeurones.train()
vecteur_sortie_neurones = reseauNeurones.predict()

print(repr(vecteur_sortie_neurones))