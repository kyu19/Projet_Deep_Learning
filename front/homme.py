import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import operator
import plotly.graph_objs as go

#laisser le choix du modèle à l'utilisateur
model_name = st.selectbox("Choisir un modèle", ("Modèle 1(93%)", "Modèle 2(80%)", "Modèle 3(75%)"))
if model_name == "Modèle 1(93%)":
    model = tf.keras.models.load_model("./modals/modele_91.h5")
elif model_name == "Modèle 2(80%)":
    model = tf.keras.models.load_model("./modals/modele_82.h5")
elif model_name == "Modèle 3(75%)":
    model = tf.keras.models.load_model("./modals/modele_74.h5")
else:
    #mettre un message d'erreur
    st.write("Erreur: Modèle non trouvé")

# Charger le modèle
# model = tf.keras.models.load_model("./modals/modele_91.h5")

def preprocess_image(image):
    # Redimensionner l'image à la taille attendue par le modèle
    image = image.resize((100, 100))
    # Convertir l'image en tableau numpy
    image = np.array(image)
    # Normaliser les valeurs de pixels de l'image
    image = image / 255.0
    # Ajouter une dimension pour correspondre à la forme d'entrée du modèle
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    # Prétraiter l'image
    processed_image = preprocess_image(image)
    # Faire la prédiction avec le modèle
    prediction = model.predict(processed_image)
    return prediction

def main():
    st.title("Fruit Detection")

    st.markdown("""
    Drag & drop images of fruits here🥭🍉🍊🍌🍒
    """)
    uploaded_files = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    train_dir = "../archive/fruits-360_dataset/fruits-360/Training"

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Afficher l'image téléchargée
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            # Convertir l'image téléchargée en objet Image
            pil_image = Image.open(uploaded_file)
            # Prédire à partir de l'image téléchargée
            prediction = predict(pil_image)
            class_names = os.listdir(train_dir)
            dico = {}
            for i, names in enumerate(class_names):
                dico[names] = prediction[0][i]

            #afficher le plus probable
            cle_max = max(dico.items(), key=operator.itemgetter(1))[0]
            st.write(cle_max)
            # Trouver les cinq classes avec les probabilités maximales
            top_classes = sorted(dico.items(), key=operator.itemgetter(1), reverse=True)[:5]

            # Créer des listes séparées pour les noms de classe et les probabilités
            class_names, probabilities = zip(*top_classes)

            # Afficher le graphique Plotly
            fig = go.Figure([go.Bar(x=class_names, y=probabilities)])
            fig.update_layout(title="Top 5 Predictions",
                              xaxis_title="Fruit Class",
                              yaxis_title="Probability")
            st.plotly_chart(fig)

    # Images de fruits dans la barre latérale
    st.sidebar.title("Fruit Images")
    st.sidebar.image("./style/pomme.png", caption='Apple', use_column_width=True)
    st.sidebar.image("./style/bannane.png", caption='Banana', use_column_width=True)
    st.sidebar.image("./style/orange.png", caption='Orange', use_column_width=True)

if __name__ == "__main__":
    main()
