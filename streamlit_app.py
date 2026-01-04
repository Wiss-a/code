"""
SYSTÈME DE DÉTECTION DE FRAUDE BANCAIRE - VERSION CORRIGÉE
============================================================
Corrections apportées:
1. Initialisation correcte de final_decision
2. Logique cohérente pour afficher le résultat final
3. Utilisation de final_decision au lieu de fraud_prob pour le verdict
"""

import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# CONFIGURATION PAGE
# =============================================================================
st.set_page_config(
    page_title="Détection Fraude Bancaire",
    page_icon="🔍",
    layout="wide"
)

# =============================================================================
# CSS PERSONNALISÉ
# =============================================================================
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #2c3e50;
    margin-bottom: 0.5rem;
}
.sub-header {
    text-align: center;
    color: #7f8c8d;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}
.alert-fraud {
    padding: 2rem;
    background: linear-gradient(135deg, #e74c3c, #c0392b);
    color: white;
    border-radius: 15px;
    font-size: 2rem;
    font-weight: bold;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 8px 16px rgba(231, 76, 60, 0.3);
    animation: pulse 2s infinite;
}
.alert-warning {
    padding: 2rem;
    background: linear-gradient(135deg, #f39c12, #e67e22);
    color: white;
    border-radius: 15px;
    font-size: 2rem;
    font-weight: bold;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 8px 16px rgba(243, 156, 18, 0.3);
}
.alert-safe {
    padding: 2rem;
    background: linear-gradient(135deg, #27ae60, #229954);
    color: white;
    border-radius: 15px;
    font-size: 2rem;
    font-weight: bold;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 8px 16px rgba(39, 174, 96, 0.3);
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.02); }
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CHARGEMENT DES MODÈLES
# =============================================================================
@st.cache_resource
def load_models():
    try:
        model = joblib.load('outputs/best_model.pkl')
        scaler = joblib.load('outputs/scaler.pkl')
        try:
            with open('outputs/metadata.json', 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {'best_model':'XGBoost','optimal_threshold':0.2,'all_models':{}}
        return model, scaler, metadata, metadata.get('optimal_threshold',0.2), None
    except Exception as e:
        return None, None, None, None, str(e)

model, scaler, metadata, optimal_threshold, error = load_models()

# =============================================================================
# HEADER
# =============================================================================

st.markdown('<h1 class="main-header">🔍 Système de Détection de Fraude Bancaire</h1>', 
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyse en Temps Réel avec Intelligence Artificielle | Projet CDDA 2024-2025</p>', 
            unsafe_allow_html=True)

if error:
    st.error(f"""
    ❌ **Erreur de chargement des modèles**
    
    {error}
    
    **Vérifiez que les fichiers suivants existent:**
    - `outputs/best_model.pkl`
    - `outputs/scaler.pkl`
    - `outputs/metadata.json`
    """)
    st.stop()

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header("📊 Informations du Modèle")

if metadata:
    st.sidebar.success(f"**Modèle Actif:** {metadata.get('best_model', 'XGBoost')}")
    st.sidebar.info(f"**Seuil Optimal:** {optimal_threshold:.3f}")
    
    if 'all_models' in metadata and metadata['all_models']:
        best_model_name = metadata.get('best_model', list(metadata['all_models'].keys())[0])
        if best_model_name in metadata['all_models']:
            metrics = metadata['all_models'][best_model_name]['metrics']
            
            st.sidebar.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
            st.sidebar.metric("F1-Score", f"{metrics.get('f1_score', 0)*100:.1f}%")
            st.sidebar.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")

st.sidebar.markdown("---")

# Initialiser session state pour le mode démo
if 'demo_type' not in st.session_state:
    st.session_state.demo_type = None

# Mode de démonstration
demo_mode = st.sidebar.checkbox(
    "🎮 Mode Démonstration",
    help="Remplit automatiquement avec des exemples"
)

st.sidebar.markdown("---")

# =============================================================================
# TABS PRINCIPALES
# =============================================================================

tab1, tab2, tab3 = st.tabs([
    "🔍 Analyse Transaction",
    "📊 Analyse Batch (CSV)",
    "📈 Statistiques"
])

# =============================================================================
# TAB 1: ANALYSE TRANSACTION UNIQUE (VERSION CORRIGÉE)
# =============================================================================

with tab1:
    st.header("Analyse d'une Transaction Individuelle")
    
    # Exemples prédéfinis avec boutons
    if demo_mode:
        st.info("🎮 **Mode Démonstration Activé** - Choisissez un exemple")
        
        col_demo1, col_demo2, col_demo3 = st.columns(3)
        
        with col_demo1:
            if st.button("✅ Transaction Légitime", use_container_width=True):
                st.session_state.demo_type = "legitimate"
                st.rerun()
        
        with col_demo2:
            if st.button("⚠️ Transaction Suspecte", use_container_width=True):
                st.session_state.demo_type = "suspicious"
                st.rerun()
        
        with col_demo3:
            if st.button("🚨 Fraude Évidente", use_container_width=True):
                st.session_state.demo_type = "fraud"
                st.rerun()
    
    st.markdown("---")
    
    # Définir les valeurs par défaut AVANT de créer les widgets
    default_values = {
        'legitimate': {
            'amount': 150.0,
            'old_orig': 5000.0,
            'new_orig': 4850.0,
            'old_dest': 3000.0,
            'new_dest': 3150.0,
            'type': 'PAYMENT',
            'type_idx': 0,
            'hour': 14,
            'day': 'Mercredi',
            'day_idx': 2
        },
        'suspicious': {
            'amount': 15000.0,
            'old_orig': 20000.0,
            'new_orig': 5000.0,
            'old_dest': 5000.0,
            'new_dest': 20000.0,
            'type': 'TRANSFER',
            'type_idx': 1,
            'hour': 22,
            'day': 'Samedi',
            'day_idx': 5
        },
        'fraud': {
            'amount': 50000.0,
            'old_orig': 100.0,
            'new_orig': 0.0,
            'old_dest': 200000.0,
            'new_dest': 250000.0,
            'type': 'CASH_OUT',
            'type_idx': 2,
            'hour': 3,
            'day': 'Dimanche',
            'day_idx': 6
        }
    }
    
    # Récupérer les valeurs par défaut selon le mode démo
    current_demo = st.session_state.get('demo_type', 'legitimate')
    if not demo_mode:
        current_demo = 'legitimate'
    
    defaults = default_values.get(current_demo, default_values['legitimate'])
    
    # Afficher quel exemple est chargé
    if demo_mode and st.session_state.demo_type:
        demo_labels = {
            'legitimate': '✅ Exemple: Transaction Légitime',
            'suspicious': '⚠️ Exemple: Transaction Suspecte',
            'fraud': '🚨 Exemple: Fraude Évidente'
        }
        st.success(demo_labels[st.session_state.demo_type])
    
    # Formulaire de transaction avec KEY UNIQUE pour chaque widget
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Informations Transaction")
        
        amount = st.number_input(
            "💵 Montant de la transaction (€)",
            min_value=0.0,
            max_value=1000000.0,
            value=defaults['amount'],
            step=10.0,
            key=f"amount_{current_demo}",
            help="Montant en euros"
        )
        
        transaction_type = st.selectbox(
            "🏦 Type de transaction",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
            index=defaults['type_idx'],
            key=f"type_{current_demo}",
            help="Nature de la transaction"
        )
        
        old_balance_orig = st.number_input(
            "💼 Solde initial émetteur (€)",
            min_value=0.0,
            value=defaults['old_orig'],
            step=100.0,
            key=f"old_orig_{current_demo}"
        )
        
        new_balance_orig = st.number_input(
            "💼 Nouveau solde émetteur (€)",
            min_value=0.0,
            value=defaults['new_orig'],
            step=100.0,
            key=f"new_orig_{current_demo}"
        )
    
    with col2:
        st.subheader("👤 Informations Destinataire")
        
        old_balance_dest = st.number_input(
            "💰 Solde initial destinataire (€)",
            min_value=0.0,
            value=defaults['old_dest'],
            step=100.0,
            key=f"old_dest_{current_demo}"
        )
        
        new_balance_dest = st.number_input(
            "💰 Nouveau solde destinataire (€)",
            min_value=0.0,
            value=defaults['new_dest'],
            step=100.0,
            key=f"new_dest_{current_demo}"
        )
        
        hour = st.slider(
            "🕐 Heure de la transaction",
            0, 23,
            defaults['hour'],
            key=f"hour_{current_demo}"
        )
        
        day = st.selectbox(
            "📅 Jour de la semaine",
            ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"],
            index=defaults['day_idx'],
            key=f"day_{current_demo}"
        )
    
    st.markdown("---")
    
    # Bouton d'analyse
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_button = st.button(
            "🔍 ANALYSER LA TRANSACTION",
            type="primary",
            use_container_width=True
        )
    
    if analyze_button:
        st.markdown("---")
        st.markdown("## 🔬 DIAGNOSTIC COMPLET")
        
        # ===================================================================
        # 1. CONSTRUCTION DES FEATURES
        # ===================================================================
        st.subheader("1️⃣ Construction du Vecteur de Features")
        
        # Encoder le type
        type_encoding = {
            'PAYMENT': 1, 
            'TRANSFER': 2, 
            'CASH_OUT': 3, 
            'DEBIT': 4, 
            'CASH_IN': 5
        }
        type_encoded = type_encoding.get(transaction_type, 0)
        
        # Features dérivées
        delta_orig = old_balance_orig - new_balance_orig
        delta_dest = new_balance_dest - old_balance_dest
        ratio_amount_orig = amount / (old_balance_orig + 1e-5)  # éviter division par 0

        # Construire features finales
        features = np.array([[ 
            1,                      # step
            type_encoded,           # type
            amount,                 # amount
            old_balance_orig,       # oldbalanceOrg
            new_balance_orig,       # newbalanceOrig
            old_balance_dest,       # oldbalanceDest
            new_balance_dest       # newbalanceDest
        ]])

        # ===================================================================
        # CORRECTION CRITIQUE: Initialiser final_decision AVANT de l'utiliser
        # ===================================================================
        final_decision = 0  # Par défaut: pas de fraude évidente
        fraud_evidence_reasons = []  # Pour tracer les raisons
        
        # Détection de fraude "évidente" par règles métier
        if abs(delta_orig - amount) > 0.01:
            fraud_evidence_reasons.append(f"Δ solde émetteur ({delta_orig:.2f}€) ≠ montant transaction ({amount:.2f}€)")
            final_decision = 1
            
        if ratio_amount_orig > 10:
            fraud_evidence_reasons.append(f"Ratio montant/solde initial = {ratio_amount_orig:.1f}x (> 10x)")
            final_decision = 1
            
        if transaction_type == 'CASH_OUT' and amount > 10000:
            fraud_evidence_reasons.append(f"CASH_OUT de {amount:,.0f}€ (> 10,000€)")
            final_decision = 1
        
        # Afficher l'alerte de fraude évidente si détectée
        if final_decision == 1:
            st.error("🚨 **FRAUDE ÉVIDENTE DÉTECTÉE par règles métiers**")
            st.warning("**Raisons:**")
            for reason in fraud_evidence_reasons:
                st.write(f"- {reason}")

        # Afficher les features BRUTES
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Features BRUTES:**")
            df_raw = pd.DataFrame({
                'Feature': ['step', 'type', 'amount', 'oldbalanceOrg', 
                           'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'],
                'Valeur': features[0]
            })
            st.dataframe(df_raw, use_container_width=True)
        
        with col2:
            st.write("**Informations:**")
            st.metric("Type Transaction", f"{transaction_type} (code: {type_encoded})")
            st.metric("Montant", f"{amount:,.2f} €")
            st.metric("Δ Solde Émetteur", f"{delta_orig:,.2f} €")
            st.metric("Δ Solde Destinataire", f"{delta_dest:,.2f} €")
        
        # ===================================================================
        # 2. SCALING
        # ===================================================================
        st.markdown("---")
        st.subheader("2️⃣ Application du Scaling")
        
        try:
            scaled_data = scaler.transform(features)
            st.success("✅ Scaling appliqué avec succès")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Features APRÈS Scaling:**")
                df_scaled = pd.DataFrame({
                    'Feature': ['step', 'type', 'amount', 'oldbalanceOrg', 
                               'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'],
                    'Valeur Scalée': scaled_data[0]
                })
                st.dataframe(df_scaled, use_container_width=True)
            
            with col2:
                st.write("**Statistiques du Scaling:**")
                st.write(f"Min: {scaled_data[0].min():.4f}")
                st.write(f"Max: {scaled_data[0].max():.4f}")
                st.write(f"Mean: {scaled_data[0].mean():.4f}")
                st.write(f"Std: {scaled_data[0].std():.4f}")
                
        except Exception as e:
            st.error(f"❌ Erreur lors du scaling: {str(e)}")
            st.stop()
        
        # ===================================================================
        # 3. PRÉDICTION BRUTE
        # ===================================================================
        st.markdown("---")
        st.subheader("3️⃣ Prédiction du Modèle")
        
        try:
            # Probabilités
            probabilities = model.predict_proba(scaled_data)[0]
            fraud_prob = float(probabilities[1])
            legit_prob = float(probabilities[0])
            
            # Prédiction binaire avec différents seuils
            pred_050 = 1 if fraud_prob >= 0.50 else 0
            pred_077 = 1 if fraud_prob >= 0.77 else 0
            pred_030 = 1 if fraud_prob >= 0.30 else 0
            
            st.success("✅ Prédiction réussie")
            
            # Affichage des probabilités
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Probabilité FRAUDE",
                    f"{fraud_prob*100:.2f}%",
                    delta=f"{(fraud_prob - 0.2)*100:+.1f}% vs seuil 0.2"
                )
            
            with col2:
                st.metric(
                    "Probabilité LÉGITIME",
                    f"{legit_prob*100:.2f}%"
                )
            
            with col3:
                st.metric(
                    "Confiance",
                    f"{max(probabilities)*100:.2f}%"
                )
            
            # Tableau de décision selon les seuils
            st.write("**Décision selon différents seuils:**")
            decision_df = pd.DataFrame({
                'Seuil': ['0.30 (Sensible)', '0.50 (Standard)', '0.77 (Training Optimal)'],
                'Probabilité Fraude': [f"{fraud_prob*100:.2f}%"] * 3,
                'Décision Modèle': [
                    '🚨 FRAUDE' if pred_030 == 1 else '✅ LÉGITIME',
                    '🚨 FRAUDE' if pred_050 == 1 else '✅ LÉGITIME',
                    '🚨 FRAUDE' if pred_077 == 1 else '✅ LÉGITIME'
                ],
                'Dépasse Seuil?': [
                    '✅ OUI' if fraud_prob >= 0.30 else '❌ NON',
                    '✅ OUI' if fraud_prob >= 0.50 else '❌ NON',
                    '✅ OUI' if fraud_prob >= 0.77 else '❌ NON'
                ]
            })
            st.dataframe(decision_df, use_container_width=True)
            
            # ===================================================================
            # 4. ANALYSE DES FEATURES IMPORTANTES
            # ===================================================================
            st.markdown("---")
            st.subheader("4️⃣ Analyse des Features")
            
            # Vérifier si le modèle a feature_importances_
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = ['step', 'type', 'amount', 'oldbalanceOrg', 
                               'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances,
                    'Valeur Brute': features[0],
                    'Valeur Scalée': scaled_data[0]
                }).sort_values('Importance', ascending=False)
                
                st.write("**Importance des Features (selon le modèle):**")
                st.dataframe(importance_df, use_container_width=True)
                
                # Graphique
                fig = px.bar(
                    importance_df, 
                    x='Feature', 
                    y='Importance',
                    title='Importance des Features dans le Modèle'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # ===================================================================
            # 5. VÉRIFICATIONS DE COHÉRENCE
            # ===================================================================
            st.markdown("---")
            st.subheader("5️⃣ Vérifications de Cohérence")
            
            checks = []
            
            # Check 1: Cohérence des soldes
            if abs(delta_orig - amount) > 0.01:
                checks.append({
                    'Check': 'Cohérence Solde Émetteur',
                    'Status': '⚠️ INCOHÉRENT',
                    'Détail': f'Δ solde ({delta_orig:.2f}) ≠ montant ({amount:.2f})'
                })
            else:
                checks.append({
                    'Check': 'Cohérence Solde Émetteur',
                    'Status': '✅ OK',
                    'Détail': f'Δ solde = montant'
                })
            
            # Check 2: Soldes négatifs
            if new_balance_orig < 0 or new_balance_dest < 0:
                checks.append({
                    'Check': 'Soldes Positifs',
                    'Status': '⚠️ SOLDE NÉGATIF',
                    'Détail': 'Un solde est négatif (suspect)'
                })
            else:
                checks.append({
                    'Check': 'Soldes Positifs',
                    'Status': '✅ OK',
                    'Détail': 'Tous les soldes sont positifs'
                })
            
            # Check 3: Transaction suspecte
            if amount > old_balance_orig * 1.5:
                checks.append({
                    'Check': 'Montant vs Solde',
                    'Status': '⚠️ SUSPECT',
                    'Détail': f'Montant ({amount:.0f}€) > 150% du solde initial'
                })
            else:
                checks.append({
                    'Check': 'Montant vs Solde',
                    'Status': '✅ OK',
                    'Détail': 'Montant cohérent avec le solde'
                })
            
            # Check 4: Type de transaction
            if transaction_type in ['CASH_OUT', 'TRANSFER'] and amount > 10000:
                checks.append({
                    'Check': 'Type & Montant',
                    'Status': '⚠️ RISQUE ÉLEVÉ',
                    'Détail': f'{transaction_type} de {amount:,.0f}€ (suspect)'
                })
            else:
                checks.append({
                    'Check': 'Type & Montant',
                    'Status': '✅ OK',
                    'Détail': 'Combinaison normale'
                })
            
            checks_df = pd.DataFrame(checks)
            st.dataframe(checks_df, use_container_width=True)
            
           # ===================================================================
            # 6. RÉSULTAT FINAL (VERSION ADAPTÉE POUR SMOTE)
            # ===================================================================
            st.markdown("---")
            st.markdown("## 🎯 RÉSULTAT FINAL")

            # Seuils adaptatifs pour modèle SMOTE
            THRESHOLD_CONSERVATIVE = 0.70
            THRESHOLD_BALANCED = 0.50
            THRESHOLD_AGGRESSIVE = 0.30

            # Utiliser le seuil du metadata, ou BALANCED par défaut
            decision_threshold = metadata.get('recommended_thresholds', {}).get('balanced', THRESHOLD_BALANCED)

            # Afficher une note explicative
            st.info("""
            📊 **Note sur les probabilités:**
            Le modèle a été entraîné sur des données équilibrées (40% fraudes).
            Les probabilités affichées sont **relatives** et indiquent un **score de risque**.
            """)

            if final_decision == 1:
                # Fraude évidente par règles métier
                st.markdown('<div class="alert-fraud">🚨 ALERTE FRAUDE DÉTECTÉE 🚨</div>', unsafe_allow_html=True)
                st.error(f"""
                **Fraude détectée par les RÈGLES MÉTIER**
                
                Anomalies critiques détectées indépendamment du modèle.
                """)
            elif fraud_prob >= THRESHOLD_BALANCED:
                # Fraude détectée par le modèle
                st.markdown('<div class="alert-fraud">🚨 ALERTE FRAUDE DÉTECTÉE 🚨</div>', unsafe_allow_html=True)
                st.error(f"""
                **Fraude détectée par le MODÈLE ML**
                
                Score de risque: {fraud_prob*100:.2f}%
                Seuil de décision: {decision_threshold*100:.0f}%
                
                ⚠️ Ce score est relatif et indique une forte probabilité de fraude.
                """)
            elif fraud_prob >= THRESHOLD_AGGRESSIVE:
                # Transaction suspecte
                st.markdown('<div class="alert-warning">⚠️ TRANSACTION SUSPECTE</div>', unsafe_allow_html=True)
                st.warning(f"""
                **Transaction nécessitant une vérification**
                
                Score de risque: {fraud_prob*100:.2f}%
                """)
            else:
                # Transaction légitime
                st.markdown('<div class="alert-safe">✅ TRANSACTION LÉGITIME</div>', unsafe_allow_html=True)
                st.success(f"""
                **Transaction approuvée**
                
                Score de risque: {fraud_prob*100:.2f}% (faible)
                """)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
            st.exception(e)
