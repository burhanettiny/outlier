import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Interlaboratory Comparison Tool", layout="wide")
st.title("🔬 Interlaboratory Comparison & Outlier Detection")

st.write("👉 Copy your data from Excel and paste it below (Ctrl+V).")

# --- Paste Excel data
pasted_data = st.text_area("Paste here:", height=200)

df = None
if pasted_data:
    try:
        df = pd.read_csv(io.StringIO(pasted_data), sep="\t")
        st.success("✅ Data parsed successfully!")
    except Exception as e:
        st.error(f"Failed to read data: {e}")

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

    # --- Outlier detection method suggestion
    st.sidebar.header("📌 Outlier Detection Options")
    try:
        stat, p_val = stats.normaltest(df[x_col])
        if p_val > 0.05:
            method_suggestion = "Data is normally distributed → Grubbs test is suitable."
        else:
            method_suggestion = "Data is non-normal or heterogeneous → Modified Z-score is recommended."
    except:
        method_suggestion = "Z-score or Modified Z-score can be used."
    st.sidebar.info(f"Suggestion: {method_suggestion}")

    methods = st.sidebar.multiselect(
        "Select methods:",
        ["Z-score", "Modified Z-score", "Grubbs test"],
        default=["Z-score"]
    )

    results = df.copy()
    outlier_suggestions = []

    # --- Apply methods
    if "Z-score" in methods:
        results["zscore"] = (results[x_col] - consensus) / results[u_col]
        results["outlier_z"] = np.abs(results["zscore"]) > 2
        for i, row in results.iterrows():
            if row["outlier_z"]:
                outlier_suggestions.append(
                    f"Lab {i+1}: Z-score = {row['zscore']:.2f} → This measurement exceeds 2σ limit, possible measurement error or lab deviation."
                )

    if "Modified Z-score" in methods:
        median_x = np.median(results[x_col])
        mad = np.median(np.abs(results[x_col] - median_x))
        if mad == 0:
            mad = 1e-6
        results["modz"] = 0.6745 * (results[x_col] - median_x) / mad
        results["outlier_modz"] = np.abs(results["modz"]) > 3.5
        for i, row in results.iterrows():
            if row["outlier_modz"]:
                outlier_suggestions.append(
                    f"Lab {i+1}: Modified Z-score = {row['modz']:.2f} → This measurement deviates more than 3.5 MAD from the median, caution advised."
                )

    if "Grubbs test" in methods:
        try:
            data = results[x_col].copy()
            n = len(data)
            grubbs_outliers = []
            while n > 2:
                mean_x = np.mean(data)
                std_x = np.std(data, ddof=1)
                diff = np.abs(data - mean_x)
                max_idx = diff.idxmax()
                G = diff[max_idx] / std_x
                crit = ((n-1)/np.sqrt(n)) * np.sqrt(stats.t.ppf(1-0.05/(2*n), n-2)**2 / (n-2 + stats.t.ppf(1-0.05/(2*n), n-2)**2))
                if G > crit:
                    lab_num = max_idx + 1
                    grubbs_outliers.append((lab_num, G, crit))
                    data = data.drop(max_idx)
                    n = len(data)
                else:
                    break
            results["outlier_grubbs"] = results.index.map(lambda i: any(i+1==o[0] for o in grubbs_outliers))
            for lab_num, G_val, crit_val in grubbs_outliers:
                outlier_suggestions.append(
                    f"Lab {lab_num}: Grubbs G = {G_val:.4f} > critical {crit_val:.4f} → Extreme value, considered outlier."
                )
        except Exception as e:
            st.warning(f"Grubbs test could not be executed: {e}")

    st.subheader("✅ Results")
    st.dataframe(results, use_container_width=True)

    st.subheader("ℹ️ Outlier Detection Suggestions")
    if outlier_suggestions:
        for s in outlier_suggestions:
            st.markdown(f"- {s}")
    else:
        st.markdown("No outliers detected. Data mostly aligns with the consensus.")

    # --- Plot with lab IDs and colored outliers
    st.subheader("📈 Visualization")
    fig, ax = plt.subplots(figsize=(9, 4))
    lab_ids = np.arange(1, len(results) + 1)

    # Outlier mask
    outlier_mask = results.get("outlier_z", False) | results.get("outlier_modz", False) | results.get("outlier_grubbs", False)

    # Normal and outlier points
    ax.errorbar(lab_ids[~outlier_mask], results[x_col][~outlier_mask],
                yerr=results[u_col][~outlier_mask], fmt='o', color='blue', label='Normal')
    ax.errorbar(lab_ids[outlier_mask], results[x_col][outlier_mask],
                yerr=results[u_col][outlier_mask], fmt='o', color='red', label='Outlier')

    ax.axhline(consensus, color='green', linestyle='--', label="Consensus")
    ax.axhspan(ci95_low, ci95_high, color='green', alpha=0.2, label="95% CI")
    ax.set_xlabel("Lab ID")
    ax.set_xticks(lab_ids)
    ax.set_ylabel("Measured value")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    # --- Residual plot vs NIST CE consensus
    st.subheader("📊 Deviation from Consensus (Residuals)")
    fig2, ax2 = plt.subplots(figsize=(9,4))

    residuals = results[x_col] - consensus

    ax2.bar(lab_ids[~outlier_mask], residuals[~outlier_mask], color='blue', label='Normal')
    ax2.bar(lab_ids[outlier_mask], residuals[outlier_mask], color='red', label='Outlier')

    ax2.axhline(0, color='green', linestyle='--', label="Consensus")
    ax2.axhspan(ci95_low - consensus, ci95_high - consensus, color='green', alpha=0.2, label="95% CI")
    ax2.set_xlabel("Lab ID")
    ax2.set_ylabel("Deviation from Consensus")
    ax2.set_xticks(lab_ids)
    ax2.legend(fontsize=8)
    st.pyplot(fig2)
