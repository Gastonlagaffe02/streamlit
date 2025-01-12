import streamlit as st
import pandas as pd
import joblib
import numpy as np
def run():
    # Charger les modèles préalablement sauvegardés
    lasso_model = joblib.load('models/lasso_model.pkl')
    svm_model1 = joblib.load('models/svm_model1_linear.pkl')

    # Interface utilisateur Streamlit
    st.title("Prédiction du Risque de Santé")

    # Demander à l'utilisateur d'entrer les valeurs pour les variables
    model_type = st.sidebar.selectbox("Choisissez le modèle de prédiction:", ["SVM", "Lasso"])

    if model_type == "SVM":
        st.write("Entrez les valeurs pour les variables Cardiovasculaire, Respiratoire, Rénale et Taux Décès.")
        cardio = st.number_input("Cardiovasculaire", min_value=0, max_value=100, value=0, key='cardio')
        respiratoire = st.number_input("Respiratoire", min_value=0, max_value=100, value=0, key='respiratoire')
        renale = st.number_input("Rénale", min_value=0, max_value=100, value=0, key='renale')
        taux_deces = st.number_input("Taux Décès", min_value=0.0, max_value=100.0, value=0.0, key='taux_deces')

        if st.button("Prédire", key='predict_button_svm'):
            # Créer un DataFrame avec les valeurs entrées par l'utilisateur
            user_input = pd.DataFrame({
                'Respiratoire': [respiratoire],
                'Cardiovasculaire': [cardio],
                'Rénale': [renale],
                'Taux Décès': [taux_deces]
            })

            # Faire la prédiction avec le modèle SVM
            prediction = svm_model1.predict(user_input)

            # Afficher la classe prédite
            risk_levels = {0: 'Faible', 1: 'Moyenne', 2: 'Élevée', 3: 'Très Élevée'}
            predicted_risk = risk_levels[prediction[0]]

            st.write(f"La classe de risque prédite est : {predicted_risk}")

            # Afficher des recommandations en fonction de la classe prédite
            if prediction[0] == 0:
                st.write("Risque faible. Aucune action immédiate nécessaire.")
            elif prediction[0] == 1:
                st.write("Risque modéré. Il est conseillé de suivre un mode de vie plus sain.")
            elif prediction[0] == 2:
                st.write("Risque élevé. Consultez un professionnel de santé.")
            else:
                st.write("Risque très élevé. Une évaluation médicale immédiate est recommandée.")

    elif model_type == "Lasso":
        st.write("Entrez les valeurs pour les variables Cardiovasculaire, Respiratoire et Rénale.")
        cardio = st.number_input("Cardiovasculaire", min_value=0, max_value=100, value=0, key='cardio_lasso')
        respiratoire = st.number_input("Respiratoire", min_value=0, max_value=100, value=0, key='respiratoire_lasso')
        renale = st.number_input("Rénale", min_value=0, max_value=100, value=0, key='renale_lasso')

        if st.button("Prédire", key='predict_button_lasso'):
            # Créer un DataFrame avec les valeurs entrées par l'utilisateur
            user_input = pd.DataFrame({
                'Cardiovasculaire': [cardio],
                'Respiratoire': [respiratoire],
                'Rénale': [renale]
            })

            # Faire la prédiction avec le modèle Lasso
            prediction = lasso_model.predict(user_input)

            # Afficher la prédiction
            st.write(f"Le taux de décès prédit est : {prediction[0]:.2f}")
