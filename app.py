import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Assuming your backend is in the src/ folder ---
try:
    from src.agent_executor import OptimizationExecutor
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    st.warning("Backend modules (src.agent_executor) not found. Running in UI-only / Mock mode.")

st.set_page_config(page_title="Quant Portfolio Optimizer", layout="wide", page_icon="📈")

# ==========================================
# 1. DUMMY DATA GENERATOR
# ==========================================
@st.cache_data
def generate_dummy_data(n_assets=200):
    np.random.seed(42)
    
    # 1. Base Universe (Equities, Bonds, Derivatives)
    n_standard = n_assets - 3
    tickers = [f"TICKER_{i:03d}" for i in range(n_standard)]
    types = np.random.choice(["EQUITY", "BOND", "DERIVATIVE"], size=n_standard, p=[0.6, 0.3, 0.1])
    currencies = np.random.choice(["USD", "EUR", "GBP"], size=n_standard, p=[0.7, 0.2, 0.1])
    sectors = np.random.choice(["Tech", "Healthcare", "Financials", "Energy", "Consumer"], size=n_standard)
    
    df = pd.DataFrame({
        "Ticker": tickers,
        "InstrumentType": types,
        "Currency": currencies,
        "Sector": sectors,
        "Expected_Return": np.random.normal(0.05, 0.08, n_standard),
        "ESG_Score": np.random.randint(20, 100, n_standard)
    })
    
    # 2. Add Cash Buckets
    cash_data = pd.DataFrame({
        "Ticker": ["CASH_USD", "CASH_EUR", "CASH_GBP"],
        "InstrumentType": ["CASH", "CASH", "CASH"],
        "Currency": ["USD", "EUR", "GBP"],
        "Sector": ["Cash", "Cash", "Cash"],
        "Expected_Return": [0.02, 0.01, 0.015],
        "ESG_Score": [50, 50, 50]
    })
    df = pd.concat([df, cash_data], ignore_index=True)
    
    # 3. Add Financial Attributes
    df["Cash_Impact"] = np.where(df["InstrumentType"] == "DERIVATIVE", 0.0, 1.0)
    df["Exposure"] = 1.0
    
    fx_rates = {"USD": 1.0, "EUR": 1.1, "GBP": 1.25}
    df["FXRateToBase"] = df["Currency"].map(fx_rates)
    
    # 4. Generate Covariance Matrix (Positive Semi-Definite)
    n_total = len(df)
    factor = np.random.randn(n_total, 5)
    cov_matrix = np.dot(factor, factor.T) * 0.01
    np.fill_diagonal(cov_matrix, cov_matrix.diagonal() + 0.02) # Ensure strict pos-def
    
    # 5. Generate Initial Portfolio (w0) & Benchmark
    w0 = np.random.uniform(0, 1, n_total)
    w0[df["InstrumentType"] == "DERIVATIVE"] = 0.0 # No initial derivatives for simplicity
    w0 = w0 / np.sum(w0)
    
    benchmark = np.random.uniform(0, 1, n_total)
    benchmark = benchmark / np.sum(benchmark)
    
    matrix_inputs = {
        "Variance": cov_matrix.tolist(),
        "w0": w0.tolist(),
        "Benchmark": benchmark.tolist()
    }
    
    return df, matrix_inputs

# ==========================================
# 2. SESSION STATE INIT
# ==========================================
if "df_data" not in st.session_state:
    st.session_state.df_data, st.session_state.matrix_inputs = generate_dummy_data()
if "opt_result" not in st.session_state:
    st.session_state.opt_result = None

# ==========================================
# 3. UI LAYOUT (TABS)
# ==========================================
st.title("📈 Institutional Portfolio Optimizer")
tab1, tab2, tab3 = st.tabs(["📊 1. Universe & Data", "⚙️ 2. Optimization Setup", "🚀 3. Results & Analytics"])

# ------------------------------------------
# TAB 1: UNIVERSE
# ------------------------------------------
with tab1:
    st.header("Universe Selection & Initial State")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Assets", len(st.session_state.df_data))
    with col2:
        st.metric("Base Currency", "USD")
    with col3:
        st.metric("Initial NAV", "$100,000,000")
        
    st.subheader("Asset Universe (Preview)")
    st.dataframe(st.session_state.df_data.head(15), use_container_width=True)
    
    st.subheader("Initial Weights vs Benchmark (Top 10)")
    df_plot = st.session_state.df_data[["Ticker", "Sector"]].copy()
    df_plot["w0"] = st.session_state.matrix_inputs["w0"]
    df_plot["Benchmark"] = st.session_state.matrix_inputs["Benchmark"]
    df_top10 = df_plot.sort_values("w0", ascending=False).head(10)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_top10["Ticker"], y=df_top10["w0"], name="Initial Weight (w0)"))
    fig.add_trace(go.Bar(x=df_top10["Ticker"], y=df_top10["Benchmark"], name="Benchmark"))
    fig.update_layout(barmode='group', title="Top Initial Positions", yaxis_tickformat='.2%')
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# TAB 2: OPTIMIZATION SETUP
# ------------------------------------------
with tab2:
    st.header("Optimization Problem Definition")
    
    mode_tab1, mode_tab2 = st.tabs(["Standard Allocation (No Cashflows)", "Execution & Cashflows"])
    
    # Helper to build constraints based on UI
    def build_constraints(enforce_long_only, enforce_fully_invested):
        c = []
        if enforce_long_only:
            c.append({"is_strict": True, "applied_to": "x", "attribute": "element", "bound_type": "min", "value": 0.0})
        if enforce_fully_invested:
            c.append({"is_strict": True, "applied_to": "x", "attribute": "sum_all", "bound_type": "eq", "value": 1.0})
        return c

    # --- Mode 1: Standard ---
    with mode_tab1:
        st.subheader("Standard Objective")
        std_obj = st.selectbox("Objective", ["Min Variance", "Max Expected Return", "Min Tracking Error"], key="std_obj")
        
        st.subheader("Standard Constraints")
        std_long = st.checkbox("Long Only (x >= 0)", value=True, key="std_long")
        std_full = st.checkbox("Fully Invested (sum x = 1)", value=True, key="std_full")
        
        if st.button("Run Standard Optimization", type="primary", use_container_width=True):
            cfg = {
                "objective": {
                    "type": "quadratic" if std_obj == "Min Variance" else "linear" if std_obj == "Max Expected Return" else "tracking_error",
                    "target_name": "Variance" if std_obj in ["Min Variance", "Min Tracking Error"] else "Expected_Return",
                    "direction": "min" if std_obj in ["Min Variance", "Min Tracking Error"] else "max"
                },
                "constraints": build_constraints(std_long, std_full)
            }
            with st.spinner("Running CasADi Solver..."):
                if BACKEND_AVAILABLE:
                    executor = OptimizationExecutor(verbose=True)
                    st.session_state.opt_result = executor.run(cfg, st.session_state.matrix_inputs, st.session_state.df_data)
                else:
                    st.success("Mock Run Successful! (Connect backend for real results)")
            st.rerun()

    # --- Mode 2: Cashflow ---
    with mode_tab2:
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Cashflow Parameters")
            nav0 = st.number_input("Current NAV ($)", value=100000000.0, step=1000000.0)
            cf_amount = st.number_input("Cashflow Amount (Inflow > 0, Outflow < 0)", value=5000000.0, step=500000.0)
            tc_rate = st.number_input("Transaction Cost Rate (bps)", value=10.0, step=1.0) / 10000.0
            
        with colB:
            st.subheader("Objectives & Constraints")
            cf_obj = st.selectbox("Objective", ["Min Variance", "Min Tracking Error", "Min Turnover"], key="cf_obj")
            
            st.markdown("**Trade Constraints**")
            max_trades = st.number_input("Max Number of Trades (Cardinality)", min_value=0, value=20, step=1)
            min_ticket = st.number_input("Min Ticket Size ($)", min_value=0.0, value=50000.0, step=10000.0)

        if st.button("Run Cashflow Optimization (MIP)", type="primary", use_container_width=True):
            cfg = {
                "cashflow": {
                    "nav0": nav0,
                    "amount": cf_amount,
                    "base_currency": "USD",
                    "tc_enabled": True,
                    "tc_rate": tc_rate,
                    "use_foreign_cash": True
                },
                "objective": {
                    "type": "quadratic" if cf_obj == "Min Variance" else "tracking_error" if cf_obj == "Min Tracking Error" else "turnover",
                    "target_name": "Variance",
                    "direction": "min"
                },
                "constraints": build_constraints(True, False) # Sum=1 handled by cashflow equations natively
            }
            
            # Add Binary constraints for execution
            if max_trades > 0:
                cfg["constraints"].append({"is_strict": True, "applied_to": "b", "constraint_family": "cardinality", "attribute": "sum_all", "bound_type": "max", "value": max_trades})
            if min_ticket > 0:
                cfg["constraints"].append({"is_strict": True, "applied_to": "b", "constraint_family": "min_ticket", "min_value": min_ticket / (nav0 + cf_amount)})

            with st.spinner("Running CasADi MIP Solver..."):
                if BACKEND_AVAILABLE:
                    executor = OptimizationExecutor(verbose=True)
                    st.session_state.opt_result = executor.run(cfg, st.session_state.matrix_inputs, st.session_state.df_data)
                else:
                    st.success("Mock Cashflow Run Successful! (Connect backend for real results)")
            st.rerun()

# ------------------------------------------
# TAB 3: RESULTS & ANALYTICS
# ------------------------------------------
with tab3:
    res = st.session_state.opt_result
    if res is None:
        st.info("No optimization results yet. Please run an optimization in Tab 2.")
    else:
        status_color = "green" if res.get("success") else "red"
        st.markdown(f"### Optimization Status: :{status_color}[{res.get('status', 'Unknown')}]")
        
        if res.get("success"):
            # Prepare result dataframe
            df_res = st.session_state.df_data[["Ticker", "InstrumentType", "Currency", "Sector"]].copy()
            df_res["w0"] = st.session_state.matrix_inputs["w0"]
            df_res["w_final"] = res.get("weights_final", res.get("x_values", []))
            df_res["w_diff"] = df_res["w_final"] - df_res["w0"]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Objective Value", f"{res.get('objective', 0.0):.6f}")
            col2.metric("Turnover", f"{np.sum(np.abs(df_res['w_diff'])):.2%}")
            
            is_cf = "nav1" in res
            if is_cf:
                col3.metric("Final NAV", f"${res.get('nav1', 0):,.0f}")
                col4.metric("Total TC Paid", f"${res.get('tc_total', 0):,.0f}")
            
            st.divider()
            
            row1_col1, row1_col2 = st.columns(2)
            
            with row1_col1:
                st.subheader("Sector Allocation Shift")
                sector_agg = df_res.groupby("Sector")[["w0", "w_final"]].sum().reset_index()
                fig = px.bar(sector_agg, x="Sector", y=["w0", "w_final"], barmode="group", title="Initial vs Final by Sector")
                fig.update_layout(yaxis_tickformat='.2%')
                st.plotly_chart(fig, use_container_width=True)
                
            with row1_col2:
                if is_cf and "trades_net" in res:
                    st.subheader("Executed Trades (EUR/USD Nominals)")
                    # Filter active trades
                    df_res["Trade_Net"] = res.get("trades_full_vector_pre_cost", [0]*len(df_res))
                    active_trades = df_res[df_res["Trade_Net"] != 0].copy()
                    active_trades.sort_values("Trade_Net", inplace=True)
                    
                    fig2 = px.bar(active_trades, x="Trade_Net", y="Ticker", orientation='h', 
                                  color="Trade_Net", color_continuous_scale="RdYlGn",
                                  title=f"Net Trades Generated (Count: {len(active_trades)})")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.subheader("Top Active Over/Underweights")
                    df_res_sorted = df_res.sort_values("w_diff", ascending=False)
                    top_diff = pd.concat([df_res_sorted.head(5), df_res_sorted.tail(5)])
                    fig2 = px.bar(top_diff, x="w_diff", y="Ticker", orientation='h', title="Top Weight Changes")
                    fig2.update_layout(xaxis_tickformat='.2%')
                    st.plotly_chart(fig2, use_container_width=True)
            
            st.subheader("Final Allocation Table")
            st.dataframe(df_res.style.format({
                "w0": "{:.2%}", "w_final": "{:.2%}", "w_diff": "{:.2%}", "Trade_Net": "${:,.0f}"
            }), use_container_width=True)