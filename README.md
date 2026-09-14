import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

# Import de ton routeur institutionnel
from src.agent_executor import OptimizationExecutor

# ==========================================
# 0. CONFIGURATION ET CACHE (MOCK DATA)
# ==========================================
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

@st.cache_data
def generate_mock_data(n_assets=20):
    """Générateur de bouchon : à remplacer par des requêtes SQL/API plus tard."""
    np.random.seed(42)
    tickers = [f"TICK{i:02d}" for i in range(1, n_assets+1)]
    secteurs = ["Tech", "Finance", "Energy", "Healthcare"] * (n_assets // 4)
    esg_scores = np.random.uniform(40, 95, n_assets)
    expected_returns = np.random.uniform(0.02, 0.15, n_assets)
    
    df_data = pd.DataFrame({
        "Ticker": tickers,
        "Sector": secteurs[:n_assets],
        "ESG_Score": esg_scores,
        "Expected_Return": expected_returns
    })
    
    # Matrice de covariance semi-définie positive
    A = np.random.randn(n_assets, n_assets)
    sigma = np.dot(A, A.T) / 100 
    
    matrix_inputs = {
        "Variance": sigma.tolist(),
        "w0": np.full(n_assets, 1.0/n_assets).tolist(),
        "Benchmark": np.full(n_assets, 1.0/n_assets).tolist()
    }
    
    return df_data, matrix_inputs

df_data, matrix_inputs = generate_mock_data(20)

# Initialisation des variables d'état (Session State)
if "objectives" not in st.session_state:
    # Objectif par défaut pour éviter un lancement à vide
    st.session_state["objectives"] = [{"type": "quadratic", "target_name": "Variance", "direction": "min", "name": "Risk"}]
if "constraints" not in st.session_state:
    # Le budget à 100% est indispensable
    st.session_state["constraints"] = [{"applied_to": "x", "attribute": "sum_all", "is_strict": True, "bound_type": "eq", "value": 1.0}]

# ==========================================
# TITRE ET ONGLETS
# ==========================================
st.title("⚙️ Moteur d'Optimisation de Portefeuille (Hybride)")
st.markdown("Interface de test du pipeline **CasADi (Déterministe)** & **Pymoo (Génétique)**.")

tab1, tab2, tab3 = st.tabs(["1️⃣ Univers & Inputs", "2️⃣ Objectifs & Contraintes", "3️⃣ Exécution & Résultats"])

# ==========================================
# ONGLET 1 : UNIVERS ET POIDS
# ==========================================
with tab1:
    st.header("Données de Marché")
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Univers d'investissement")
        st.dataframe(df_data, use_container_width=True, height=250)
        
    with colB:
        st.subheader("Poids Initiaux (w0)")
        st.info("Ici, tu pourras uploader un CSV contenant les poids initiaux pour le calcul du Turnover. Pour la démo, ils sont équipondérés.")
        uploaded_file = st.file_uploader("Upload CSV w0 (Optionnel)", type=["csv"])
        if uploaded_file is not None:
            st.success("Fichier chargé ! (Logique de parsing à implémenter)")
            
    st.subheader("Matrice de Covariance (Aperçu)")
    st.dataframe(pd.DataFrame(matrix_inputs["Variance"], columns=df_data["Ticker"], index=df_data["Ticker"]).iloc[:5, :5])

# ==========================================
# ONGLET 2 : CONFIGURATEUR
# ==========================================
with tab2:
    col_obj, col_cstr = st.columns(2)
    
    # --- GESTION DES OBJECTIFS ---
    with col_obj:
        st.header("🎯 Objectifs")
        
        with st.form("add_obj_form", clear_on_submit=True):
            obj_target = st.selectbox("Métrique", ["Variance", "Expected_Return", "ESG_Score", "Tracking_Error"])
            obj_dir = st.radio("Direction", ["min", "max"], horizontal=True)
            
            if st.form_submit_button("Ajouter l'Objectif"):
                obj_type = "quadratic" if obj_target in ["Variance", "Tracking_Error"] else "linear"
                st.session_state["objectives"].append({
                    "type": obj_type,
                    "target_name": obj_target,
                    "direction": obj_dir,
                    "name": f"{obj_dir.capitalize()} {obj_target}"
                })
                st.rerun()
                
        st.write("**Objectifs actuels :**")
        for i, obj in enumerate(st.session_state["objectives"]):
            st.markdown(f"- **{obj['name']}** ({obj['type']})")
            
        if st.button("Vider les Objectifs"):
            st.session_state["objectives"] = []
            st.rerun()

    # --- GESTION DES CONTRAINTES ---
    with col_cstr:
        st.header("🧱 Contraintes")
        
        with st.form("add_cstr_form", clear_on_submit=True):
            c_attr = st.selectbox("Attribut", ["sum_all", "element", "min_buy_in", "Sector", "ESG_Score", "Turnover"])
            c_type = st.selectbox("Type", ["eq", "max", "min"])
            c_val = st.number_input("Valeur", value=1.0, format="%.4f")
            
            # Options avancées
            c_strict = st.checkbox("Strict (Hard Constraint)", value=True)
            c_apply = st.radio("Appliquer à", ["x (Poids)", "b (Sélection binaire)"], horizontal=True)
            
            c_target = ""
            if c_attr == "Sector":
                c_target = st.selectbox("Secteur cible", df_data["Sector"].unique())
            
            if st.form_submit_button("Ajouter la Contrainte"):
                applied_to = "b" if "b" in c_apply else "x"
                fam = "cardinality" if applied_to == "b" else ("min_buy_in" if c_attr == "min_buy_in" else "standard")
                
                new_cstr = {
                    "applied_to": applied_to,
                    "attribute": c_attr,
                    "constraint_family": fam,
                    "is_strict": c_strict,
                    "bound_type": c_type,
                    "value": c_val
                }
                if c_target:
                    new_cstr["targets"] = [c_target]
                    
                st.session_state["constraints"].append(new_cstr)
                st.rerun()
                
        st.write("**Contraintes actuelles :**")
        for i, c in enumerate(st.session_state["constraints"]):
            strict_badge = "🔴 HARD" if c["is_strict"] else "🔵 SOFT"
            target_str = f" [{c['targets'][0]}]" if "targets" in c else ""
            st.markdown(f"- {strict_badge} | {c['applied_to']} | {c['attribute']}{target_str} {c['bound_type']} {c['value']}")
            
        if st.button("Vider les Contraintes (sauf Budget)"):
            st.session_state["constraints"] = [{"applied_to": "x", "attribute": "sum_all", "is_strict": True, "bound_type": "eq", "value": 1.0}]
            st.rerun()

# ==========================================
# ONGLET 3 : EXÉCUTION & RÉSULTATS
# ==========================================
with tab3:
    st.header("🚀 Lancement du Moteur")
    
    # Construction du JSON final tel que l'attend ton backend
    vars_mixte = [
        {"name": "x", "size": "n_rows", "type": "continuous"},
        {"name": "b", "size": "n_rows", "type": "binary"}
    ]
    
    optimization_config = {
        "decision_variables": vars_mixte,
        "objectives": st.session_state["objectives"],
        "constraints": st.session_state["constraints"],
        "moo": {"pop_size": 50, "n_gen": 100, "seed": 42, "verbose": False}
    }
    
    with st.expander("Voir le Payload JSON envoyé à l'Agent Executor"):
        st.json(optimization_config)
        
    if st.button("🔥 LANCER L'OPTIMISATION", type="primary", use_container_width=True):
        if not st.session_state["objectives"]:
            st.error("Ajoute au moins un objectif !")
            st.stop()
            
        with st.spinner("Le moteur quantitatif tourne (Routage automatique CasADi / Pymoo)..."):
            # INSTANCIATION DE TON ROUTEUR EXACT
            executor = OptimizationExecutor(verbose=False)
            try:
                res = executor.run(optimization_config, matrix_inputs, df_data)
            except Exception as e:
                st.error(f"Erreur d'exécution : {str(e)}")
                st.stop()
                
        st.success(f"Optimisation terminée ! Statut : {res.get('status', 'OK')}")
        
        # --- CAS 1 : MULTI-OBJECTIF (Front de Pareto) ---
        if res.get("type") == "pareto":
            st.subheader(f"Front de Pareto ({res.get('algorithm')})")
            pts = res.get("pareto_points", [])
            st.metric("Portefeuilles viables générés", len(pts))
            
            if len(pts) > 0:
                # Extraction des données pour Plotly
                obj_names = pts[0]["objective_names"]
                n_obj = len(obj_names)
                
                # Création d'un DataFrame de résultats
                res_data = []
                for idx, p in enumerate(pts):
                    row = {"ID": idx}
                    for i, name in enumerate(obj_names):
                        # On ré-inverse pour l'affichage si c'était une maximisation
                        val = p["objectives_minimised"][i]
                        row[name] = -val if st.session_state["objectives"][i]["direction"] == "max" else val
                    res_data.append(row)
                    
                df_res = pd.DataFrame(res_data)
                
                # Affichage graphique dynamique
                col_graph, col_pt = st.columns([2, 1])
                
                with col_graph:
                    if n_obj == 2:
                        fig = px.scatter(df_res, x=obj_names[0], y=obj_names[1], text="ID", 
                                         title="Frontière Efficiente", template="plotly_white")
                        fig.update_traces(marker=dict(size=10, color="blue", opacity=0.7), textposition="top center")
                        st.plotly_chart(fig, use_container_width=True)
                    elif n_obj == 3:
                        fig = px.scatter_3d(df_res, x=obj_names[0], y=obj_names[1], z=obj_names[2],
                                            color="ID", title="Frontière 3D")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Visualisation pour plus de 3 objectifs à implémenter via PCP (Parallel Coordinates).")
                        st.dataframe(df_res)
                        
                with col_pt:
                    st.write("**Inspecter un portefeuille spécifique :**")
                    pt_id = st.selectbox("Sélectionne l'ID du portefeuille (voir graphique)", df_res["ID"].tolist())
                    selected_w = pts[pt_id]["weights"]
                    
                    df_w = pd.DataFrame({"Ticker": df_data["Ticker"], "Poids": selected_w})
                    df_w = df_w[df_w["Poids"] > 1e-4].sort_values(by="Poids", ascending=False)
                    
                    fig_w = px.bar(df_w, x="Ticker", y="Poids", title=f"Poids du portefeuille {pt_id}")
                    st.plotly_chart(fig_w, use_container_width=True)
                    
        # --- CAS 2 : SINGLE-OBJECTIVE (Déterministe CasADi) ---
        else:
            st.subheader("Résultat Déterministe (CasADi)")
            st.metric("Valeur de l'objectif", round(res.get("objective", 0), 6))
            
            weights = res.get("x_values", [])
            if weights:
                df_w = pd.DataFrame({"Ticker": df_data["Ticker"], "Poids": weights})
                df_w = df_w[df_w["Poids"] > 1e-4].sort_values(by="Poids", ascending=False)
                
                fig = px.bar(df_w, x="Ticker", y="Poids", title="Composition du Portefeuille Optimal",
                             color="Poids", color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True)