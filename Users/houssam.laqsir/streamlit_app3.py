import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Détection de Fraude",
    page_icon="🔍",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    """Charge le modèle et le scaler depuis les fichiers"""
    try:
        model = joblib.load('fraud_detection_rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

# Fonction pour créer les features dérivées
def create_features(data):
    """Crée les features dérivées nécessaires au modèle"""
    df = data.copy()
    
    # Variation de balance
    df['balanceChange_orig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balanceChange_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # Ratio montant/solde
    df['amountToBalanceRatio_orig'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    
    # Indicateurs binaires
    df['isOriginEmpty'] = (df['newbalanceOrig'] == 0).astype(int)
    df['isDestEmpty'] = (df['oldbalanceDest'] == 0).astype(int)
    
    # Erreurs de balance
    df['errorBalanceOrig'] = df['balanceChange_orig'] - df['amount']
    df['errorBalanceDest'] = df['balanceChange_dest'] - df['amount']
    
    return df

# Fonction pour créer un graphique de gauge
def create_gauge(value):
    """Crée un graphique de type gauge pour la probabilité"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 50}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#2ecc71'},
                {'range': [50, 80], 'color': '#f39c12'},
                {'range': [80, 100], 'color': '#e74c3c'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# Header
st.title("🔍 Détection de Fraude - Test de Transaction")
st.markdown("### Analysez une transaction pour détecter une potentielle fraude")
st.markdown("---")

# Charger le modèle
model, scaler, error = load_model()

if error:
    st.error(f"❌ Erreur lors du chargement du modèle: {error}")
    st.info("""
    **Instructions:**
    1. Placez les fichiers suivants dans le même dossier que cette application:
       - `fraud_detection_rf_model.pkl`
       - `scaler.pkl`
    2. Relancez l'application
    """)
    st.stop()

st.success("✅ Modèle chargé avec succès!")
st.markdown("---")

# Formulaire de saisie
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 💳 Informations de la Transaction")
    
    amount = st.number_input(
        "Montant de la transaction (€)",
        min_value=0.0,
        value=10000.0,
        step=100.0,
        help="Montant de la transaction en euros"
    )
    
    transaction_type = st.selectbox(
        "Type de transaction",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"],
        help="Sélectionnez le type de transaction"
    )
    
    step = st.number_input(
        "Période (step)",
        min_value=1,
        value=1,
        help="Unité de temps de la transaction"
    )
    
    st.markdown("### 📤 Compte Origine")
    
    oldbalanceOrg = st.number_input(
        "Solde avant transaction (€)",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        key="old_orig"
    )
    
    newbalanceOrig = st.number_input(
        "Solde après transaction (€)",
        min_value=0.0,
        value=40000.0,
        step=1000.0,
        key="new_orig"
    )

with col2:
    st.markdown("### 🏦 Compte Destinataire")
    
    oldbalanceDest = st.number_input(
        "Solde avant transaction (€)",
        min_value=0.0,
        value=20000.0,
        step=1000.0,
        key="old_dest"
    )
    
    newbalanceDest = st.number_input(
        "Solde après transaction (€)",
        min_value=0.0,
        value=30000.0,
        step=1000.0,
        key="new_dest"
    )

st.markdown("---")

# Bouton d'analyse
if st.button("🔍 ANALYSER LA TRANSACTION", use_container_width=True):
    with st.spinner("🔄 Analyse en cours..."):
        # Créer le DataFrame avec les données
        input_data = pd.DataFrame({
            'step': [step],
            'amount': [amount],
            'oldbalanceOrg': [oldbalanceOrg],
            'newbalanceOrig': [newbalanceOrig],
            'oldbalanceDest': [oldbalanceDest],
            'newbalanceDest': [newbalanceDest],
            'type_CASH_IN': [1 if transaction_type == 'CASH_IN' else 0],
            'type_CASH_OUT': [1 if transaction_type == 'CASH_OUT' else 0],
            'type_DEBIT': [1 if transaction_type == 'DEBIT' else 0],
            'type_PAYMENT': [1 if transaction_type == 'PAYMENT' else 0],
            'type_TRANSFER': [1 if transaction_type == 'TRANSFER' else 0]
        })
        
        # Créer les features dérivées
        input_data = create_features(input_data)
        
        try:
            # Normaliser les features
            features_scaled = scaler.transform(input_data)
            
            # Prédiction
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            fraud_prob = probability[1]
            is_fraud = prediction == 1
            
            st.markdown("---")
            st.markdown("## 📋 RÉSULTAT DE L'ANALYSE")
            
            # Résultat principal
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if is_fraud:
                    st.error("# 🚨 TRANSACTION SUSPECTE !")
                else:
                    st.success("# ✅ TRANSACTION LÉGITIME")
                
                # Gauge de probabilité
                fig_gauge = create_gauge(fraud_prob)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.markdown(f"### Probabilité de fraude: **{fraud_prob*100:.2f}%**")
            
            st.markdown("---")
            
            # Métriques détaillées
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                risk_level = "🔴 ÉLEVÉ" if fraud_prob > 0.8 else ("🟠 MOYEN" if fraud_prob > 0.5 else "🟢 FAIBLE")
                st.markdown(f"""
                <div class="metric-box">
                    <h3>Niveau de Risque</h3>
                    <h2>{risk_level}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>Confiance</h3>
                    <h2>{max(probability)*100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>Type</h3>
                    <h2>{transaction_type}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>Montant</h3>
                    <h2>{amount:,.0f} €</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Indicateurs d'anomalie
        #     st.markdown("### 🔍 Indicateurs d'Anomalie Détectés")
            
        #     balance_change_orig = oldbalanceOrg - newbalanceOrig
        #     balance_change_dest = newbalanceDest - oldbalanceDest
            
        #     anomalies = []
            
        #     if abs(balance_change_orig - amount) > 0.01:
        #         anomalies.append("⚠️ **Incohérence dans le solde du compte origine**")
            
        #     if abs(balance_change_dest - amount) > 0.01:
        #         anomalies.append("⚠️ **Incohérence dans le solde du compte destinataire**")
            
        #     if newbalanceOrig == 0 and oldbalanceOrg > 0:
        #         anomalies.append("🔴 **Le compte origine a été complètement vidé**")
            
        #     if amount / (oldbalanceOrg + 1) > 0.9:
        #         anomalies.append("🔴 **La transaction représente plus de 90% du solde**")
            
        #     if oldbalanceDest == 0:
        #         anomalies.append("⚠️ **Le compte destinataire avait un solde nul**")
            
        #     if len(anomalies) > 0:
        #         for anomaly in anomalies:
        #             st.warning(anomaly)
        #     else:
        #         st.info("✅ Aucune anomalie majeure détectée dans les soldes")
            
        #     st.markdown("---")
            
        #     # Recommandations
        #     st.markdown("### 💡 Recommandations")
            
        #     if fraud_prob > 0.8:
        #         st.error("""
        #         **🚨 ACTIONS URGENTES REQUISES:**
        #         - 🚫 **Bloquer immédiatement la transaction**
        #         - 📞 **Contacter le client pour vérification d'identité**
        #         - 🔒 **Geler temporairement le compte**
        #         - 📝 **Créer un rapport d'incident détaillé**
        #         - 👮 **Envisager d'informer les autorités si confirmé**
        #         """)
        #     elif fraud_prob > 0.5:
        #         st.warning("""
        #         **⚠️ ACTIONS DE VÉRIFICATION:**
        #         - ⏸️ **Mettre la transaction en attente**
        #         - ✅ **Demander une vérification d'identité secondaire**
        #         - 📧 **Envoyer une notification au client**
        #         - 📊 **Surveiller l'activité du compte pendant 24h**
        #         """)
        #     else:
        #         st.success("""
        #         **✅ TRANSACTION APPROUVÉE:**
        #         - ✅ **Autoriser la transaction**
        #         - 📊 **Enregistrer dans les logs de routine**
        #         - 📈 **Continuer la surveillance normale**
        #         """)
            
        #     # Détails techniques (optionnel, en expander)
        #     with st.expander("🔧 Voir les détails techniques"):
        #         st.markdown("**Features calculées:**")
                
        #         details = {
        #             "Variation solde origine": f"{balance_change_orig:,.2f} €",
        #             "Variation solde destination": f"{balance_change_dest:,.2f} €",
        #             "Ratio montant/solde origine": f"{amount / (oldbalanceOrg + 1):.4f}",
        #             "Compte origine vidé": "Oui" if newbalanceOrig == 0 else "Non",
        #             "Destination avec solde nul": "Oui" if oldbalanceDest == 0 else "Non",
        #             "Erreur balance origine": f"{balance_change_orig - amount:,.2f} €",
        #             "Erreur balance destination": f"{balance_change_dest - amount:,.2f} €"
        #         }
                
        #         for key, value in details.items():
        #             st.text(f"{key}: {value}")
        
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
            st.info("Vérifiez que toutes les valeurs sont correctes et réessayez.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🤖 Modèle: Random Forest | 📊 Précision: 99.6% | 🎯 F1-Score: 0.75</p>
    <p>Développé avec Azure Machine Learning & Streamlit</p>
</div>
""", unsafe_allow_html=True)