# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 16:16:50 2023

@author: Asma
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import *
from tkinter import filedialog
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_image():
    print("Séléctionner l'image à partir du disque: ")
    filename = filedialog.askopenfilename()
    img = load_img(filename, target_size=(224, 224))
   
    img = img_to_array(img)
    img = img.reshape(1,224,224,3)
    #img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    
    return img



def run(model):
    #load the image
    img= load_image()

    pred = model.predict(img)
    return pred



# Charger le modèle InceptionV3 pré entraîné et ajusté finement
model = tf.keras.models.load_model("C:/Users/Hanoun/Desktop/openclassrooms/projet 6/model_classification_race_chien_v2.h5")

# Définir le fichier contenant les classes des résultats
file_name = "list_race_name.json"
with open(file_name) as json_file:
    race_list = json.load(json_file)
    

# entry point, run the example
pred = run(model)
val_pred = np.argmax(pred,axis=1)
race = race_list[str(val_pred.item())]
proba = pred[0][val_pred] *100
print('La race du chien est: %s avec une probabilité de  (%.2f%%)' % (race, proba))


