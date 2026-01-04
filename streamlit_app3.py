import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Configuration de la page
st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    try:
        model = joblib.load('fraud_detection_rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Fichiers du modèle introuvables. Assurez-vous que 'fraud_detection_rf_model.pkl' et 'scaler.pkl' sont dans le même répertoire.")
        return None, None

# Fonction pour charger les données de prédiction existantes
@st.cache_data
def load_prediction_data():
    try:
        df = pd.read_csv('fraud_predictions_powerbi.csv')
        return df
    except FileNotFoundError:
        return None

# Fonction de feature engineering
def engineer_features(data):
    """Applique le même feature engineering que dans le notebook"""
    df = data.copy()
    
    # Créer les features dérivées
    df['balanceChange_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balanceChange_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['amountToBalanceRatio_orig'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['isOriginEmpty'] = (df['newbalanceOrig'] == 0).astype(int)
    df['isDestEmpty'] = (df['oldbalanceDest'] == 0).astype(int)
    df['errorBalanceOrig'] = df['balanceChange_orig'] - df['amount']
    df['errorBalanceDest'] = df['balanceChange_dest'] - df['amount']
    
    return df

# Fonction de prédiction
def predict_fraud(model, scaler, input_data):
    """Effectue une prédiction de fraude"""
    # Feature engineering
    input_df = engineer_features(input_data)
    
    # Sélectionner les features dans le bon ordre
    feature_columns = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest',
        'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 
        'type_PAYMENT', 'type_TRANSFER',
        'balanceChange_orig', 'balanceChange_dest',
        'amountToBalanceRatio_orig', 'isOriginEmpty', 
        'isDestEmpty', 'errorBalanceOrig', 'errorBalanceDest'
    ]
    
    X = input_df[feature_columns]
    
    # Normalisation
    X_scaled = scaler.transform(X)
    
    # Prédiction
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    
    return prediction, probability

# Titre principal
st.markdown("""
    <h1 style='text-align: center; color: #1f77b4;'>
        🛡️ Système de Détection de Fraude Bancaire
    </h1>
    <p style='text-align: center; color: #666;'>
        Utilisant un modèle Random Forest pour détecter les transactions frauduleuses
    </p>
    <hr>
""", unsafe_allow_html=True)

# Chargement du modèle
model, scaler = load_model()

if model is None or scaler is None:
    st.stop()

# Sidebar - Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Sélectionnez une page",
    ["🏠 Accueil", "🔍 Prédiction Unique", "📁 Prédiction par Lot", "📈 Analytics & Visualisations"]
)

# PAGE 1: ACCUEIL
if page == "🏠 Accueil":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>🎯 Précision</h3>
                <h2>99.63%</h2>
                <p>Taux de prédictions correctes</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3>🎪 F1-Score</h3>
                <h2>75.00%</h2>
                <p>Équilibre précision/rappel</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3>📊 ROC-AUC</h3>
                <h2>99.75%</h2>
                <p>Performance globale</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 À propos du modèle")
        st.info("""
        **Algorithme**: Random Forest (100 arbres)
        
        **Dataset d'entraînement**: 
        - 6.3M transactions
        - 8,213 cas de fraude (0.13%)
        - Ratio 773:1 (très déséquilibré)
        
        **Techniques utilisées**:
        - Feature Engineering avancé
        - SMOTE + RandomUnderSampling
        - Validation croisée
        - Optimisation des hyperparamètres
        """)
    
    with col2:
        st.markdown("### 🎯 Caractéristiques principales")
        st.success("""
        **Top 5 Features Importantes**:
        1. 💰 Nouveau solde origine
        2. 📊 Ratio montant/solde
        3. 🔄 Variation solde origine
        4. 🏦 Ancien solde destination
        5. 💳 Type de paiement
        
        **Performance**:
        - ✅ Détecte 100% des fraudes (Recall)
        - ✅ Minimal de faux positifs
        - ✅ Temps de prédiction < 1ms
        """)
    
    st.markdown("---")
    st.markdown("### 🚀 Comment utiliser cette application")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔍 Prédiction Unique**
        
        Analysez une transaction individuelle en saisissant manuellement les détails.
        Idéal pour les vérifications ponctuelles.
        """)
    
    with col2:
        st.markdown("""
        **📁 Prédiction par Lot**
        
        Uploadez un fichier CSV contenant plusieurs transactions pour une analyse en masse.
        """)
    
    with col3:
        st.markdown("""
        **📈 Analytics**
        
        Explorez les visualisations et statistiques des prédictions précédentes.
        """)

# PAGE 2: PRÉDICTION UNIQUE
elif page == "🔍 Prédiction Unique":
    st.header("🔍 Analyse d'une Transaction")
    st.markdown("Saisissez les détails de la transaction à analyser :")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Informations de base")
        
        step = st.number_input(
            "Étape temporelle",
            min_value=1,
            max_value=1000,
            value=1,
            help="Moment de la transaction (1-1000)"
        )
        
        transaction_type = st.selectbox(
            "Type de transaction",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
        )
        
        amount = st.number_input(
            "Montant (€)",
            min_value=0.0,
            max_value=10000000.0,
            value=1000.0,
            step=100.0,
            format="%.2f"
        )
    
    with col2:
        st.subheader("💰 Soldes du compte émetteur")
        
        oldbalance_org = st.number_input(
            "Ancien solde origine (€)",
            min_value=0.0,
            max_value=100000000.0,
            value=5000.0,
            step=100.0,
            format="%.2f"
        )
        
        newbalance_org = st.number_input(
            "Nouveau solde origine (€)",
            min_value=0.0,
            max_value=100000000.0,
            value=4000.0,
            step=100.0,
            format="%.2f"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🏦 Soldes du compte destinataire")
        
        oldbalance_dest = st.number_input(
            "Ancien solde destination (€)",
            min_value=0.0,
            max_value=100000000.0,
            value=1000.0,
            step=100.0,
            format="%.2f"
        )
    
    with col4:
        st.write("")
        st.write("")
        newbalance_dest = st.number_input(
            "Nouveau solde destination (€)",
            min_value=0.0,
            max_value=100000000.0,
            value=2000.0,
            step=100.0,
            format="%.2f"
        )
    
    st.markdown("---")
    
    if st.button("🔮 Analyser la Transaction", type="primary", use_container_width=True):
        # Créer le DataFrame d'entrée
        input_data = pd.DataFrame({
            'step': [step],
            'amount': [amount],
            'oldbalanceOrg': [oldbalance_org],
            'newbalanceOrig': [newbalance_org],
            'oldbalanceDest': [oldbalance_dest],
            'newbalanceDest': [newbalance_dest],
            'type_CASH_IN': [1 if transaction_type == 'CASH_IN' else 0],
            'type_CASH_OUT': [1 if transaction_type == 'CASH_OUT' else 0],
            'type_DEBIT': [1 if transaction_type == 'DEBIT' else 0],
            'type_PAYMENT': [1 if transaction_type == 'PAYMENT' else 0],
            'type_TRANSFER': [1 if transaction_type == 'TRANSFER' else 0]
        })
        
        # Prédiction
        with st.spinner("Analyse en cours..."):
            prediction, probability = predict_fraud(model, scaler, input_data)
        
        st.markdown("---")
        st.subheader("📊 Résultats de l'Analyse")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Jauge de risque
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Niveau de Risque de Fraude", 'font': {'size': 24}},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#90EE90'},
                        {'range': [30, 70], 'color': '#FFD700'},
                        {'range': [70, 100], 'color': '#FF6B6B'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Verdict
        if prediction == 1:
            st.error(f"""
            ### 🚨 ALERTE FRAUDE DÉTECTÉE
            
            **Probabilité de fraude**: {probability*100:.2f}%
            
            **Recommandations**:
            - ⚠️ Bloquer la transaction immédiatement
            - 📞 Contacter le client pour vérification
            - 🔍 Lancer une enquête approfondie
            - 📝 Documenter l'incident
            """)
        else:
            if probability > 0.3:
                st.warning(f"""
                ### ⚠️ Transaction Suspecte
                
                **Probabilité de fraude**: {probability*100:.2f}%
                
                **Recommandations**:
                - 🔍 Surveillance accrue recommandée
                - 📊 Vérifier l'historique du compte
                - ✅ Autoriser avec vigilance
                """)
            else:
                st.success(f"""
                ### ✅ Transaction Légitime
                
                **Probabilité de fraude**: {probability*100:.2f}%
                
                **Statut**: Transaction considérée comme sûre
                """)
        
        # Détails de la transaction
        st.markdown("---")
        st.subheader("📋 Détails de la Transaction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Montant", f"{amount:,.2f} €")
            st.metric("Type", transaction_type)
            st.metric("Ancien solde origine", f"{oldbalance_org:,.2f} €")
        
        with col2:
            balance_change = oldbalance_org - newbalance_org
            st.metric("Variation solde", f"{balance_change:,.2f} €", delta=f"{-balance_change:,.2f} €")
            ratio = amount / (oldbalance_org + 1)
            st.metric("Ratio montant/solde", f"{ratio:.2%}")
            is_empty = "Oui" if newbalance_org == 0 else "Non"
            st.metric("Compte vidé ?", is_empty)

# PAGE 3: PRÉDICTION PAR LOT
elif page == "📁 Prédiction par Lot":
    st.header("📁 Analyse par Lot")
    st.markdown("Uploadez un fichier CSV contenant plusieurs transactions à analyser.")
    
    # Template téléchargeable
    st.subheader("📥 Template CSV")
    
    template_data = pd.DataFrame({
        'step': [1, 2, 3],
        'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT'],
        'amount': [1000.0, 5000.0, 2000.0],
        'oldbalanceOrg': [5000.0, 10000.0, 2000.0],
        'newbalanceOrig': [4000.0, 5000.0, 0.0],
        'oldbalanceDest': [1000.0, 2000.0, 5000.0],
        'newbalanceDest': [2000.0, 7000.0, 7000.0]
    })
    
    st.download_button(
        label="📥 Télécharger le template CSV",
        data=template_data.to_csv(index=False),
        file_name="template_transactions.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Upload de fichier
    uploaded_file = st.file_uploader(
        "Choisissez un fichier CSV",
        type=['csv'],
        help="Le fichier doit contenir les colonnes: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest"
    )
    
    if uploaded_file is not None:
        try:
            # Charger le CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Fichier chargé avec succès: {len(df)} transactions")
            
            # Afficher un aperçu
            st.subheader("👀 Aperçu des données")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Vérifier les colonnes requises
            required_columns = ['step', 'type', 'amount', 'oldbalanceOrg', 
                              'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Colonnes manquantes: {', '.join(missing_columns)}")
            else:
                if st.button("🔮 Analyser toutes les transactions", type="primary"):
                    with st.spinner("Analyse en cours..."):
                        # Encoder le type de transaction
                        df_processed = pd.get_dummies(df, columns=['type'], prefix='type')
                        
                        # Ajouter les colonnes manquantes si nécessaire
                        for col in ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 
                                   'type_PAYMENT', 'type_TRANSFER']:
                            if col not in df_processed.columns:
                                df_processed[col] = 0
                        
                        # Prédictions
                        predictions = []
                        probabilities = []
                        
                        for idx in range(len(df_processed)):
                            row_data = df_processed.iloc[[idx]]
                            pred, prob = predict_fraud(model, scaler, row_data)
                            predictions.append(pred)
                            probabilities.append(prob)
                        
                        # Ajouter les résultats
                        df['Prédiction'] = ['Fraude' if p == 1 else 'Légitime' for p in predictions]
                        df['Probabilité_Fraude'] = [f"{p*100:.2f}%" for p in probabilities]
                        df['Niveau_Risque'] = [
                            'Élevé' if p >= 0.7 else ('Moyen' if p >= 0.3 else 'Faible')
                            for p in probabilities
                        ]
                    
                    st.success("✅ Analyse terminée!")
                    
                    # Statistiques
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_frauds = sum(predictions)
                        st.metric("Fraudes Détectées", total_frauds)
                    
                    with col2:
                        fraud_rate = (total_frauds / len(predictions)) * 100
                        st.metric("Taux de Fraude", f"{fraud_rate:.2f}%")
                    
                    with col3:
                        high_risk = sum([1 for p in probabilities if p >= 0.7])
                        st.metric("Risque Élevé", high_risk)
                    
                    with col4:
                        total_amount = df[df['Prédiction'] == 'Fraude']['amount'].sum()
                        st.metric("Montant Frauduleux", f"{total_amount:,.0f} €")
                    
                    # Graphiques
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribution des prédictions
                        fig = px.pie(
                            values=[len(df) - total_frauds, total_frauds],
                            names=['Légitime', 'Fraude'],
                            title="Distribution des Prédictions",
                            color_discrete_sequence=['#90EE90', '#FF6B6B']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Distribution des niveaux de risque
                        risk_counts = df['Niveau_Risque'].value_counts()
                        fig = px.bar(
                            x=risk_counts.index,
                            y=risk_counts.values,
                            title="Transactions par Niveau de Risque",
                            labels={'x': 'Niveau de Risque', 'y': 'Nombre de Transactions'},
                            color=risk_counts.index,
                            color_discrete_map={
                                'Faible': '#90EE90',
                                'Moyen': '#FFD700',
                                'Élevé': '#FF6B6B'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Tableau des résultats
                    st.markdown("---")
                    st.subheader("📊 Résultats Détaillés")
                    
                    # Filtres
                    col1, col2 = st.columns(2)
                    with col1:
                        filter_prediction = st.multiselect(
                            "Filtrer par prédiction",
                            options=['Légitime', 'Fraude'],
                            default=['Légitime', 'Fraude']
                        )
                    with col2:
                        filter_risk = st.multiselect(
                            "Filtrer par niveau de risque",
                            options=['Faible', 'Moyen', 'Élevé'],
                            default=['Faible', 'Moyen', 'Élevé']
                        )
                    
                    df_filtered = df[
                        (df['Prédiction'].isin(filter_prediction)) &
                        (df['Niveau_Risque'].isin(filter_risk))
                    ]
                    
                    st.dataframe(df_filtered, use_container_width=True)
                    
                    # Export
                    st.markdown("---")
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger les résultats (CSV)",
                        data=csv,
                        file_name="resultats_predictions.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")

# PAGE 4: ANALYTICS
elif page == "📈 Analytics & Visualisations":
    st.header("📈 Analytics & Visualisations")
    
    # Charger les données existantes
    prediction_data = load_prediction_data()
    
    if prediction_data is not None:
        st.success(f"✅ Données chargées: {len(prediction_data):,} transactions")
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_frauds = prediction_data['actual_fraud'].sum()
            st.metric("Total Fraudes", f"{total_frauds:,}")
        
        with col2:
            detected_frauds = prediction_data[
                (prediction_data['actual_fraud'] == True) & 
                (prediction_data['predicted_fraud'] == 1)
            ].shape[0]
            st.metric("Fraudes Détectées", f"{detected_frauds:,}")
        
        with col3:
            detection_rate = (detected_frauds / total_frauds * 100) if total_frauds > 0 else 0
            st.metric("Taux de Détection", f"{detection_rate:.1f}%")
        
        with col4:
            false_positives = prediction_data[
                (prediction_data['actual_fraud'] == False) & 
                (prediction_data['predicted_fraud'] == 1)
            ].shape[0]
            st.metric("Faux Positifs", f"{false_positives:,}")
        
        st.markdown("---")
        
        # Visualisations
        tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🎯 Performance", "💰 Montants"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution des prédictions
                pred_counts = prediction_data['prediction_category'].value_counts()
                fig = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    title="Distribution par Catégorie de Risque",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Résultats de prédiction
                result_counts = prediction_data['prediction_result'].value_counts()
                fig = px.bar(
                    x=result_counts.index,
                    y=result_counts.values,
                    title="Résultats des Prédictions",
                    labels={'x': 'Résultat', 'y': 'Nombre'},
                    color=result_counts.index,
                    color_discrete_map={
                        'Correct': '#90EE90',
                        'Faux Négatif': '#FFD700',
                        'Faux Positif': '#FF6B6B'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Distribution des probabilités
            fig = px.histogram(
                prediction_data,
                x='fraud_probability',
                nbins=50,
                title="Distribution des Probabilités de Fraude",
                labels={'fraud_probability': 'Probabilité de Fraude'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                         annotation_text="Seuil de décision")
            st.plotly_chart(fig, use_container_width=True)
            
            # Matrice de confusion
            cm_data = pd.crosstab(
                prediction_data['actual_fraud'],
                prediction_data['predicted_fraud'],
                rownames=['Réel'],
                colnames=['Prédit']
            )
            
            fig = px.imshow(
                cm_data,
                text_auto=True,
                title="Matrice de Confusion",
                labels=dict(x="Prédit", y="Réel", color="Count"),
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Montants par statut
                amount_by_fraud = prediction_data.groupby('actual_fraud')['amount'].mean()
                fig = px.bar(
                    x=['Légitime', 'Fraude'],
                    y=amount_by_fraud.values,
                    title="Montant Moyen par Statut",
                    labels={'x': 'Statut', 'y': 'Montant Moyen (€)'},
                    color=['Légitime', 'Fraude'],
                    color_discrete_map={'Légitime': '#90EE90', 'Fraude': '#FF6B6B'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribution des montants
                fig = go.Figure()
                
                fig.add_trace(go.Box(
                    y=prediction_data[prediction_data['actual_fraud'] == False]['amount'],
                    name='Légitime',
                    marker_color='#90EE90'
                ))
                
                fig.add_trace(go.Box(
                    y=prediction_data[prediction_data['actual_fraud'] == True]['amount'],
                    name='Fraude',
                    marker_color='#FF6B6B'
                ))
                
                fig.update_layout(
                    title="Distribution des Montants",
                    yaxis_title="Montant (€)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("""
        ℹ️ Aucune donnée de prédiction disponible.
        
        Pour visualiser les analytics, assurez-vous que le fichier 
        'fraud_predictions_powerbi.csv' est présent dans le répertoire.
        
        Vous pouvez générer ce fichier en:
        1. Exécutant le notebook Jupyter fourni
        2. Ou en utilisant l'ong """
        )