import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.agent_executor import OptimizationExecutor

# ==========================================
# 0. CONFIGURATION & MOCK DATA
# ==========================================
st.set_page_config(page_title="Portfolio Optimization Engine", layout="wide")

@st.cache_data
def generate_mock_data(n_assets=20):
    np.random.seed(42)
    tickers = [f"TICK{i:02d}" for i in range(1, n_assets + 1)]
    sectors = ["Technology", "Financials", "Energy", "Healthcare"] * (n_assets // 4 + 1)
    esg_scores = np.random.uniform(40, 95, n_assets)
    expected_returns = np.random.uniform(0.02, 0.15, n_assets)
    
    df_data = pd.DataFrame({
        "Ticker": tickers,
        "Sector": sectors[:n_assets],
        "ESG_Score": esg_scores,
        "Expected_Return": expected_returns
    })
    
    A = np.random.randn(n_assets, n_assets)
    sigma = np.dot(A, A.T) / 100 
    
    matrix_inputs = {
        "Variance": sigma.tolist(),
        "w0": np.full(n_assets, 1.0 / n_assets).tolist(),
        "Benchmark": np.full(n_assets, 1.0 / n_assets).tolist()
    }
    
    return df_data, matrix_inputs

df_data, default_matrix_inputs = generate_mock_data(20)

if "matrix_inputs" not in st.session_state:
    st.session_state["matrix_inputs"] = default_matrix_inputs

if "objectives" not in st.session_state:
    st.session_state["objectives"] = [{"type": "quadratic", "target_name": "Variance", "direction": "min", "name": "Minimize Risk"}]

if "constraints" not in st.session_state:
    st.session_state["constraints"] = [{"applied_to": "x", "attribute": "sum_all", "is_strict": True, "bound_type": "eq", "value": 1.0}]

if "views" not in st.session_state:
    st.session_state["views"] = []

# ==========================================
# PAGE HEADER & TABS
# ==========================================
st.title("Portfolio Optimization Engine")
st.markdown("Enterprise-grade interface for Deterministic (CasADi) and Evolutionary (Pymoo) quantitative engines.")

tab_data, tab_config, tab_exec = st.tabs(["Market Data & References", "Objectives & Constraints", "Execution & Results"])

# ==========================================
# TAB 1 : MARKET DATA & REFERENCES
# ==========================================
with tab_data:
    st.header("Investment Universe")
    st.dataframe(df_data, use_container_width=True, height=200)
    
    st.header("Reference Anchors")
    col_w0, col_bench = st.columns(2)
    
    with col_w0:
        st.subheader("Initial Weights (w0)")
        st.info("Required for Turnover constraints. CSV must contain 'Ticker' and 'Weight' columns.")
        w0_file = st.file_uploader("Upload w0 (CSV)", type=["csv"], key="w0_upload")
        if w0_file is not None:
            w0_df = pd.read_csv(w0_file)
            st.session_state["matrix_inputs"]["w0"] = w0_df["Weight"].tolist()
            st.success("Initial weights updated.")
            
    with col_bench:
        st.subheader("Benchmark Weights")
        st.info("Required for Tracking Error objectives. CSV must contain 'Ticker' and 'Weight' columns.")
        bench_file = st.file_uploader("Upload Benchmark (CSV)", type=["csv"], key="bench_upload")
        if bench_file is not None:
            bench_df = pd.read_csv(bench_file)
            st.session_state["matrix_inputs"]["Benchmark"] = bench_df["Weight"].tolist()
            st.success("Benchmark weights updated.")

# ==========================================
# TAB 2 : CONFIGURATOR
# ==========================================
with tab_config:
    col_obj, col_cstr, col_views = st.columns(3)
    
    # --- OBJECTIVES SECTION ---
    with col_obj:
        st.header("Objectives")
        with st.form("add_obj_form", clear_on_submit=True):
            obj_target = st.selectbox("Metric", ["Variance", "Expected_Return", "ESG_Score", "Tracking_Error"])
            obj_dir = st.radio("Direction", ["min", "max"], horizontal=True)
            
            if st.form_submit_button("Add Objective"):
                obj_type = "quadratic" if obj_target in ["Variance", "Tracking_Error"] else "linear"
                st.session_state["objectives"].append({
                    "type": obj_type,
                    "target_name": obj_target,
                    "direction": obj_dir,
                    "name": f"{obj_dir.capitalize()} {obj_target}"
                })
                st.rerun()
                
        st.markdown("### Active Objectives")
        for i, obj in enumerate(st.session_state["objectives"]):
            st.markdown(f"- **{obj['name']}** *(Type: {obj['type']})*")
            
        if st.button("Clear Objectives"):
            st.session_state["objectives"] = []
            st.rerun()

    # --- CONSTRAINTS SECTION ---
    with col_cstr:
        st.header("Constraints")
        with st.form("add_cstr_form", clear_on_submit=True):
            c_attr = st.selectbox("Attribute", ["sum_all", "element", "min_buy_in", "Sector", "ESG_Score", "Turnover", "exclusion"])
            
            if c_attr == "exclusion":
                c_tickers = st.multiselect("Select Assets to Mutually Exclude", df_data["Ticker"].tolist())
                c_apply = "b (Binary Selection)"
                c_strict = True
            else:
                c_target = ""
                if c_attr == "Sector":
                    c_target = st.selectbox("Target Sector", df_data["Sector"].unique())
                
                c_type = st.selectbox("Bound Type", ["eq", "max", "min", "range"])
                
                if c_type == "range":
                    c_min_val = st.number_input("Minimum Value", value=0.0, format="%.4f")
                    c_max_val = st.number_input("Maximum Value", value=1.0, format="%.4f")
                else:
                    c_val = st.number_input("Value", value=1.0, format="%.4f")
                
                c_strict = st.checkbox("Strict (Hard Constraint)", value=True)
                
                c_scale = 1.0
                if not c_strict:
                    c_scale = st.number_input("Soft Penalty Scale (Priority)", value=1.0, format="%.2f")
                    
                c_apply = st.radio("Apply To", ["x (Weights)", "b (Binary Selection)"], horizontal=True)
            
            if st.form_submit_button("Add Constraint"):
                if c_attr == "exclusion":
                    if len(c_tickers) >= 2:
                        st.session_state["constraints"].append({
                            "constraint_family": "exclusion",
                            "applied_to": "b",
                            "set": c_tickers,
                            "is_strict": True
                        })
                else:
                    applied_to = "b" if "b" in c_apply else "x"
                    fam = "cardinality" if applied_to == "b" else ("min_buy_in" if c_attr == "min_buy_in" else "standard")
                    
                    new_cstr = {
                        "applied_to": applied_to,
                        "attribute": c_attr,
                        "constraint_family": fam,
                        "is_strict": c_strict,
                        "bound_type": c_type
                    }
                    
                    if c_type == "range":
                        new_cstr["min_value"] = c_min_val
                        new_cstr["max_value"] = c_max_val
                    else:
                        new_cstr["value"] = c_val
                        
                    if not c_strict:
                        new_cstr["scale"] = c_scale
                        
                    if c_target:
                        new_cstr["targets"] = [c_target]
                        
                    st.session_state["constraints"].append(new_cstr)
                st.rerun()
                
        st.markdown("### Active Constraints")
        for i, c in enumerate(st.session_state["constraints"]):
            strict_badge = "[HARD]" if c.get("is_strict", True) else "[SOFT]"
            if c.get("constraint_family") == "exclusion":
                st.markdown(f"- {strict_badge} | Mutual Exclusion | Assets: {', '.join(c['set'])}")
            else:
                target_str = f" ({c['targets'][0]})" if "targets" in c else ""
                val_str = f"[{c.get('min_value')} to {c.get('max_value')}]" if c.get("bound_type") == "range" else c.get("value")
                scale_str = f" (Scale: {c.get('scale')})" if not c.get("is_strict", True) and "scale" in c else ""
                st.markdown(f"- {strict_badge} | {c['applied_to']} | {c['attribute']}{target_str} {c['bound_type']} {val_str}{scale_str}")
            
        if st.button("Reset Constraints"):
            st.session_state["constraints"] = [{"applied_to": "x", "attribute": "sum_all", "is_strict": True, "bound_type": "eq", "value": 1.0}]
            st.rerun()

    # --- MARKET VIEWS (BLACK-LITTERMAN) SECTION ---
    with col_views:
        st.header("Market Views")
        st.info("Input absolute return views to dynamically adjust expected returns and covariance prior to optimization.")
        with st.form("add_view_form", clear_on_submit=True):
            view_asset = st.selectbox("Target Asset", df_data["Ticker"].tolist())
            view_return = st.number_input("Expected Absolute Return (e.g., 0.15 for 15%)", value=0.05, format="%.4f")
            
            if st.form_submit_button("Add Market View"):
                st.session_state["views"].append({
                    "asset": view_asset,
                    "return": view_return
                })
                st.rerun()
                
        st.markdown("### Active Views")
        for i, v in enumerate(st.session_state["views"]):
            st.markdown(f"- **{v['asset']}**: {v['return']*100:.2f}% expected return")
            
        if st.button("Clear Views"):
            st.session_state["views"] = []
            st.rerun()

# ==========================================
# TAB 3 : EXECUTION & RESULTS
# ==========================================
with tab_exec:
    st.header("Optimization Execution")
    
    with st.expander("Evolutionary Engine Hyperparameters", expanded=False):
        col_pop, col_gen, col_seed = st.columns(3)
        pop_size = col_pop.slider("Population Size", min_value=20, max_value=200, value=100, step=10)
        n_gen = col_gen.slider("Number of Generations", min_value=50, max_value=500, value=150, step=50)
        seed = col_seed.number_input("Random Seed", value=42)

    optimization_config = {
        "decision_variables": [
            {"name": "x", "size": "n_rows", "type": "continuous"},
            {"name": "b", "size": "n_rows", "type": "binary"}
        ],
        "objectives": st.session_state["objectives"],
        "constraints": st.session_state["constraints"],
        "moo": {"pop_size": pop_size, "n_gen": n_gen, "seed": seed, "verbose": False}
    }
    
    # Translate UI views to P and Q matrices for Black-Litterman
    if st.session_state["views"]:
        n_views = len(st.session_state["views"])
        n_assets = len(df_data)
        P_matrix = np.zeros((n_views, n_assets))
        Q_vector = np.zeros(n_views)
        ticker_list = df_data["Ticker"].tolist()
        
        for i, view in enumerate(st.session_state["views"]):
            asset_idx = ticker_list.index(view["asset"])
            P_matrix[i, asset_idx] = 1.0
            Q_vector[i] = view["return"]
            
        optimization_config["black_litterman"] = {
            "P": P_matrix.tolist(),
            "Q": Q_vector.tolist(),
            "tau": 0.05,
            "rf": 0.0
        }
    
    if st.button("RUN OPTIMIZATION ENGINE", type="primary", use_container_width=True):
        if not st.session_state["objectives"]:
            st.error("Please specify at least one objective.")
            st.stop()
            
        with st.spinner("Processing... Routing engine determining optimal solver path..."):
            executor = OptimizationExecutor(verbose=False)
            try:
                res = executor.run(optimization_config, st.session_state["matrix_inputs"], df_data)
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
                st.stop()
                
        st.success(f"Execution Complete. Status: {res.get('status', 'OK')}")
        
        if res.get("type") == "pareto":
            st.subheader(f"Pareto Frontier Analysis ({res.get('algorithm')})")
            pts = res.get("pareto_points", [])
            st.metric("Feasible Portfolios Generated", len(pts))
            
            if len(pts) > 0:
                obj_names = pts[0]["objective_names"]
                soft_names = pts[0].get("soft_constraint_names", [])
                all_metrics = obj_names + soft_names
                n_metrics = len(all_metrics)
                
                res_data = []
                for idx, p in enumerate(pts):
                    row = {"ID": idx}
                    for i, name in enumerate(obj_names):
                        val = p["objectives_minimised"][i]
                        row[name] = -val if st.session_state["objectives"][i]["direction"] == "max" else val
                    for i, name in enumerate(soft_names):
                        row[name] = p["soft_losses_raw"][i]
                    res_data.append(row)
                    
                df_res = pd.DataFrame(res_data)
                
                col_graph, col_pt = st.columns([2, 1])
                
                with col_graph:
                    if n_metrics == 2:
                        fig = px.scatter(df_res, x=all_metrics[0], y=all_metrics[1], text="ID", 
                                         title="2D Efficient Frontier", template="plotly_white")
                        fig.update_traces(marker=dict(size=10, opacity=0.8), textposition="top center")
                        st.plotly_chart(fig, use_container_width=True)
                    elif n_metrics == 3:
                        fig = px.scatter_3d(df_res, x=all_metrics[0], y=all_metrics[1], z=all_metrics[2],
                                            color="ID", title="3D Efficient Frontier")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.parallel_coordinates(df_res, color="ID", dimensions=all_metrics,
                                                      title="Parallel Coordinates Plot (High-Dimensional Frontier)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                with col_pt:
                    st.markdown("**Inspect Specific Portfolio**")
                    pt_id = st.selectbox("Select Portfolio ID", df_res["ID"].tolist())
                    selected_w = pts[pt_id]["weights"]
                    
                    df_w = pd.DataFrame({"Ticker": df_data["Ticker"], "Weight": selected_w})
                    df_w = df_w[df_w["Weight"] > 1e-4].sort_values(by="Weight", ascending=False)
                    
                    fig_w = px.bar(df_w, x="Ticker", y="Weight", title=f"Composition: Portfolio {pt_id}")
                    st.plotly_chart(fig_w, use_container_width=True)
                    
        else:
            st.subheader("Deterministic Output (CasADi + SciPy MILP)")
            st.metric("Optimized Objective Value", round(res.get("objective", 0), 6))
            
            weights = res.get("x_values", [])
            if weights:
                df_w = pd.DataFrame({"Ticker": df_data["Ticker"], "Weight": weights})
                df_w = df_w[df_w["Weight"] > 1e-4].sort_values(by="Weight", ascending=False)
                
                fig = px.bar(df_w, x="Ticker", y="Weight", title="Optimal Portfolio Composition",
                             color="Weight", color_continuous_scale="Blues")
                st.plotly_chart(fig, use_container_width=True)