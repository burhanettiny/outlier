import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.title("Interlaboratory Comparison Tool (NIST CE + Outlier Detection)")

# ---------------------------
# 1. Veri yükleme veya manuel giriş
# ---------------------------
st.sidebar.header("Upload or Input Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv","xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
else:
    st.sidebar.write("Or enter data manually:")
    n_labs = st.sidebar.number_input("Number of labs", min_value=1, value=5)
    data = {"LabID": [], "x": [], "u": []}
    for i in range(1, n_labs+1):
        labid = st.sidebar.text_input(f"Lab {i} ID", f"{i}")
        x = st.sidebar.number_input(f"Lab {i} Measured Value", value=0.0, format="%.5f")
        u = st.sidebar.number_input(f"Lab {i} StdUnc", value=0.0, format="%.5f")
        data["LabID"].append(labid); data["x"].append(x); data["u"].append(u)
    df = pd.DataFrame(data)

st.subheader("Input Data")
st.dataframe(df)

# ---------------------------
# 2. NIST CE değerleri kullanıcıdan alınır
# ---------------------------
st.sidebar.header("NIST CE Parameters")
consensus = st.sidebar.number_input("Consensus estimate", value=float(df["x"].median()))
std_unc = st.sidebar.number_input("Standard uncertainty", value=0.01)
ci_low = st.sidebar.number_input("95% CI - Lower", value=consensus-0.02)
ci_high = st.sidebar.number_input("95% CI - Upper", value=consensus+0.02)

# ---------------------------
# 3. Outlier test seçenekleri
# ---------------------------
st.sidebar.header("Outlier Detection Methods")
methods = st.sidebar.multiselect(
    "Select methods:",
    ["Normalized Error (En)", "Grubbs", "Mandel h", "MAD-based"],
    default=["Normalized Error (En)"]
)

# ---------------------------
# 4. Hesaplamalar
# ---------------------------
results = df.copy()

# Normalized Error
if "Normalized Error (En)" in methods:
    def En(xi, ui, xref, uref):
        return (xi - xref) / np.sqrt(ui**2 + uref**2)
    results["En"] = results.apply(lambda r: En(r["x"], r["u"], consensus, std_unc), axis=1)
    results["En_Status"] = results["En"].apply(lambda e: "UYUMLU" if abs(e)<=1 else "UYUMSUZ")

# Grubbs (tek outlier testi)
if "Grubbs" in methods:
    G, p = stats.grubbs.test(results["x"].values, alpha=0.05)  # dikkat: scipy’de yok, ekstra paket gerekir
    # Buraya grubbs paketinden fonksiyon eklenebilir (pip install outliers)

# Mandel h / MAD-based test benzer şekilde eklenebilir...

# ---------------------------
# 5. Görselleştirme
# ---------------------------
fig, ax = plt.subplots(figsize=(8,5))
ax.errorbar(results["LabID"], results["x"], yerr=results["u"], fmt='o', color="black", capsize=4)
ax.axhline(consensus, color="red", linestyle="-", label=f"Consensus = {consensus:.3f}")
ax.axhline(ci_low, color="red", linestyle="--", label="95% CI")
ax.axhline(ci_high, color="red", linestyle="--")
ax.set_xlabel("Laboratory ID"); ax.set_ylabel("Measured value (x)")
ax.set_title("Interlaboratory Comparison - Outlier Analysis")
ax.legend()
st.pyplot(fig)

# ---------------------------
# 6. Sonuç tablosu
# ---------------------------
st.subheader("Outlier Analysis Results")
st.dataframe(results)

# ---------------------------
# 7. Recommendation
# ---------------------------
st.subheader("Recommendation")
if "En" in results:
    bad_labs = results.loc[results["En_Status"]=="UYUMSUZ","LabID"].tolist()
    if bad_labs:
        st.error(f"Uyumsuz laboratuvar(lar): {bad_labs}")
    else:
        st.success("Tüm laboratuvarlar konsensüs ile uyumlu görünüyor.")
