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

    # --- Outlier detection method suggestion with rationale
    st.sidebar.header("📌 Outlier Detection Options")
    try:
        stat, p_val = stats.normaltest(df[x_col])
        if p_val > 0.05:
            method_suggestion = "Veri normal dağılıyor → Grubbs testi uygundur."
        else:
            method_suggestion = "Veri normal değil veya heterojen → Modified Z-score önerilir."
    except:
        method_suggestion = "Z-score veya Modified Z-score kullanılabilir."
    st.sidebar.info(f"Öneri: {method_suggestion}")

    methods = st.sidebar.multiselect(
        "Select methods (with rationale):",
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
                    f"Lab {i+1}: Z-score = {row['zscore']:.2f} → Bu ölçüm 2σ sınırının dışında, olası ölçüm hatası veya laboratuvar sapması."
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
                    f"Lab {i+1}: Modified Z-score = {row['modz']:.2f} → Bu ölçüm median’dan 3.5 kat MAD sapma gösteriyor, dikkat edilmesi önerilir."
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
                crit = ( (n-1)/np.sqrt(n) ) * np.sqrt( stats.t.ppf(1-0.05/(2*n), n-2)**2 / (n-2 + stats.t.ppf(1-0.05/(2*n), n-2)**2) )
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
                    f"Lab {lab_num}: Grubbs G = {G_val:.4f} > kritik {crit_val:.4f} → En uç değer, outlier olarak değerlendirilir."
                )
        except Exception as e:
            st.warning(f"Grubbs test çalıştırılamadı: {e}")

    st.subheader("✅ Results")
    st.dataframe(results, use_container_width=True)

    st.subheader("ℹ️ Outlier Detection Suggestions")
    if outlier_suggestions:
        for s in outlier_suggestions:
            st.markdown(f"- {s}")
    else:
        st.markdown("Tespit edilen outlier yok. Veriler çoğunlukla konsensüs ile uyumlu.")

    # --- Plot with lab IDs and colored outliers
    st.subheader("📈 Visualization")
    fig, ax = plt.subplots(figsize=(9, 4))
    lab_ids = np.arange(1, len(results) + 1)

    # Outlier maskesi
    outlier_mask = results.get("outlier_z", False) | results.get("outlier_modz", False) | results.get("outlier_grubbs", False)

    # Normal ve outlier noktaları ayrı plotla
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
