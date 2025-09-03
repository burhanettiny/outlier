import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from scipy import stats

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
    st.dataframe(df, use_container_width=True)

    # --- Column selection
    st.markdown("### 🔎 Select columns for analysis")
    x_col = st.selectbox("Measured values (x)", df.columns)
    u_col = st.selectbox("Standard uncertainty (u)", df.columns)

    # --- Editable table
    st.markdown("### ✏️ Edit Data (optional)")
    df = st.data_editor(df, num_rows="dynamic")

    # --- Sidebar: NIST CE inputs
    st.sidebar.header("⚙️ NIST Consensus Parameters")
    consensus = st.sidebar.number_input(
        "Consensus estimate",
        value=float(df[x_col].median()),
        format="%.4f"
    )
    std_unc = st.sidebar.number_input(
        "Standard uncertainty",
        value=float(df[u_col].median()),
        format="%.4f"
    )
    ci95_low = st.sidebar.number_input(
        "95% CI lower",
        value=consensus - 2 * std_unc,
        format="%.4f"
    )
    ci95_high = st.sidebar.number_input(
        "95% CI upper",
        value=consensus + 2 * std_unc,
        format="%.4f"
    )

    # --- Outlier detection methods
    st.sidebar.header("📌 Outlier Detection Options")
    methods = st.sidebar.multiselect(
        "Select methods:",
        ["Z-score", "Modified Z-score", "Grubbs test"],
        default=["Z-score"]
    )

    results = df.copy()

    # --- Apply methods
    outlier_info = ""
    if "Z-score" in methods:
        results["zscore"] = (results[x_col] - consensus) / results[u_col]
        results["outlier_z"] = np.abs(results["zscore"]) > 2
        outlier_info += "- Z-score: 2’den büyük değerler outlier kabul edilir.\n"

    if "Modified Z-score" in methods:
        median_x = np.median(results[x_col])
        mad = np.median(np.abs(results[x_col] - median_x))
        if mad == 0:
            mad = 1e-6
        results["modz"] = 0.6745 * (results[x_col] - median_x) / mad
        results["outlier_modz"] = np.abs(results["modz"]) > 3.5
        outlier_info += "- Modified Z-score: 3.5’den büyük değerler outlier kabul edilir.\n"

    if "Grubbs test" in methods:
        try:
            n = len(results[x_col])
            mean_x = np.mean(results[x_col])
            std_x = np.std(results[x_col], ddof=1)
            G = np.max(np.abs(results[x_col] - mean_x)) / std_x
            results["Grubbs_G"] = G
            crit = ( (n-1)/np.sqrt(n) ) * np.sqrt( stats.t.ppf(1-0.05/(2*n), n-2)**2 / (n-2 + stats.t.ppf(1-0.05/(2*n), n-2)**2) )
            results["outlier_grubbs"] = G > crit
            outlier_info += f"- Grubbs test: G = {G:.4f}, kritik değer = {crit:.4f}. En büyük sapma outlier olarak değerlendirilir.\n"
        except Exception as e:
            st.warning(f"Grubbs test çalıştırılamadı: {e}")

    st.subheader("✅ Results")
    st.dataframe(results, use_container_width=True)

    st.subheader("ℹ️ Outlier Detection Explanation")
    st.markdown("""
    **Outlier tespiti önerisi:**  
    - Öncelikle Z-score veya Modified Z-score yöntemi ile gözlemleri kontrol edebilirsiniz.  
    - Eğer veri normal dağılıma uygunsa Grubbs testi kullanılabilir.  
    - Outlier olan veriler ölçüm hatası veya laboratuvar hatasından kaynaklanabilir ve analizden çıkarılabilir.  
    """ + outlier_info.replace("\n", "  \n"))

    # --- Plot
    st.subheader("📈 Visualization")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.errorbar(results.index, results[x_col], yerr=results[u_col], fmt='o', label="Labs")
    ax.axhline(consensus, color='green', linestyle='--', label="Consensus")
    ax.axhspan(ci95_low, ci95_high, color='green', alpha=0.2, label="95% CI")
    ax.set_xlabel("Lab ID")
    ax.set_ylabel("Measured value")
    ax.legend(fontsize=8)
    st.pyplot(fig)
