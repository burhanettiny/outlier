import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Interlaboratory Comparison Tool", layout="wide")

st.title("🔬 Interlaboratory Comparison & Outlier Detection")

st.write("👉 Excel’den verilerinizi kopyalayın ve aşağıdaki kutuya yapıştırın (Ctrl+V).")

# --- Paste Excel data
pasted_data = st.text_area("Paste here:", height=200)

df = None
if pasted_data:
    try:
        df = pd.read_csv(io.StringIO(pasted_data), sep="\t")
        st.success("✅ Data parsed successfully!")
    except Exception as e:
        st.error(f"Veri okunamadı: {e}")

if df is not None:
    st.subheader("📊 Input Data")
    st.dataframe(df)

    # --- Column selection
    st.markdown("### 🔎 Select columns for analysis")
    x_col = st.selectbox("Measured values (x)", df.columns)
    u_col = st.selectbox("Standard uncertainty (u)", df.columns)

    # --- Editable table
    st.markdown("### ✏️ Edit Data (optional)")
    df = st.data_editor(df, num_rows="dynamic")

    # --- Sidebar: NIST CE inputs
    st.sidebar.header("⚙️ NIST Consensus Parameters")
    consensus = st.sidebar.number_input("Consensus estimate", value=float(df[x_col].median()))
    std_unc = st.sidebar.number_input("Standard uncertainty", value=float(df[u_col].median()))
    ci95_low = st.sidebar.number_input("95% CI lower", value=consensus - 2 * std_unc)
    ci95_high = st.sidebar.number_input("95% CI upper", value=consensus + 2 * std_unc)

    # --- Outlier detection methods
    st.sidebar.header("📌 Outlier Detection Options")
    methods = st.sidebar.multiselect(
        "Select methods:",
        ["Z-score", "Modified Z-score", "Grubbs test"],
        default=["Z-score"]
    )

    results = df.copy()

    # --- Apply methods
    if "Z-score" in methods:
        results["zscore"] = (results[x_col] - consensus) / results[u_col]
        results["outlier_z"] = np.abs(results["zscore"]) > 2

    if "Modified Z-score" in methods:
        median_x = np.median(results[x_col])
        mad = np.median(np.abs(results[x_col] - median_x))
        if mad == 0:
            mad = 1e-6  # avoid div by zero
        results["modz"] = 0.6745 * (results[x_col] - median_x) / mad
        results["outlier_modz"] = np.abs(results["modz"]) > 3.5

    if "Grubbs test" in methods:
        from scipy import stats
        try:
            stat, p = stats.normaltest(results[x_col])
            n = len(results[x_col])
            mean_x = np.mean(results[x_col])
            std_x = np.std(results[x_col], ddof=1)
            G = np.max(np.abs(results[x_col] - mean_x)) / std_x
            results["Grubbs_G"] = G
            results["outlier_grubbs"] = G > ( (n-1)/np.sqrt(n) ) * np.sqrt( stats.t.ppf(1-0.05/(2*n), n-2)**2 / (n-2+stats.t.ppf(1-0.05/(2*n), n-2)**2) )
        except Exception as e:
            st.warning(f"Grubbs test çalıştırılamadı: {e}")

    st.subheader("✅ Results")
    st.dataframe(results)

    # --- Plot
    st.subheader("📈 Visualization")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(results.index, results[x_col], yerr=results[u_col], fmt='o', label="Labs")
    ax.axhline(consensus, color='green', linestyle='--', label="Consensus")
    ax.axhspan(ci95_low, ci95_high, color='green', alpha=0.2, label="95% CI")
    ax.set_xlabel("Lab ID")
    ax.set_ylabel("Measured value")
    ax.legend()
    st.pyplot(fig)
