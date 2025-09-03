import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Interlaboratory Comparison", layout="wide")

st.title("🔬 Interlaboratory Comparison Tool")
st.write("Excel’den verileri kopyalayıp aşağıya yapıştırın veya tabloyu doğrudan düzenleyin.")

# ----------------------------
# 1. Excel'den Paste Data
# ----------------------------
pasted_data = st.text_area("Excel tablosunu buraya yapıştırın (Ctrl+V):", height=200)

df = None
if pasted_data:
    try:
        df = pd.read_csv(io.StringIO(pasted_data), sep="\t")
    except:
        st.error("⚠️ Yapıştırılan veri işlenemedi. Lütfen satırları/sütunları kontrol edin.")

# Eğer paste yoksa örnek veri
if df is None:
    st.info("Örnek veri kullanılıyor.")
    df = pd.DataFrame({
        "LabID": [1,2,3,4,5,6,7,8,9,10,11],
        "x": [0.348,0.320,0.090,0.338,0.347,0.350,0.346,0.352,0.341,0.330,0.339],
        "u": [0.025,0.009,0.007,0.007,0.010,0.016,0.024,0.021,0.017,0.030,0.027]
    })

# ----------------------------
# 2. Data Editor (editable)
# ----------------------------
st.subheader("📋 Veri Tablosu (Düzenlenebilir)")
df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

# ----------------------------
# 3. NIST CE parametreleri
# ----------------------------
st.sidebar.header("⚙️ NIST CE Parameters")
consensus = st.sidebar.number_input("Consensus estimate", value=float(df["x"].median()))
std_unc = st.sidebar.number_input("Standard uncertainty", value=0.01, format="%.5f")
ci_low = st.sidebar.number_input("95% CI - Lower", value=consensus-0.02, format="%.5f")
ci_high = st.sidebar.number_input("95% CI - Upper", value=consensus+0.02, format="%.5f")

# ----------------------------
# 4. Outlier test seçenekleri
# ----------------------------
st.sidebar.header("🔎 Outlier Detection Methods")
methods = st.sidebar.multiselect(
    "Yöntemleri seç:",
    ["Normalized Error (En)", "MAD-based"],
    default=["Normalized Error (En)"]
)

results = df.copy()

# ----------------------------
# 5. Hesaplamalar
# ----------------------------

# Normalized Error (En)
if "Normalized Error (En)" in methods:
    def En(xi, ui, xref, uref):
        return (xi - xref) / np.sqrt(ui**2 + uref**2)
    results["En"] = results.apply(lambda r: En(r["x"], r["u"], consensus, std_unc), axis=1)
    results["En_Status"] = results["En"].apply(lambda e: "UYUMLU" if abs(e)<=1 else "UYUMSUZ")

# MAD-based outlier test
if "MAD-based" in methods:
    median_val = np.median(results["x"])
    mad = np.median(np.abs(results["x"] - median_val))
    results["MAD_score"] = abs(results["x"] - median_val) / (1.4826 * mad if mad>0 else 1e-6)
    results["MAD_Status"] = results["MAD_score"].apply(lambda s: "UYUMLU" if s<=3 else "UYUMSUZ")

# ----------------------------
# 6. Grafik
# ----------------------------
st.subheader("📊 Grafiksel Görselleştirme")
fig, ax = plt.subplots(figsize=(8,5))
ax.errorbar(results["LabID"], results["x"], yerr=results["u"], fmt='o', color="black", capsize=4)

# Konsensüs çizgileri
ax.axhline(consensus, color="red", linestyle="-", label=f"Consensus = {consensus:.3f}")
ax.axhline(ci_low, color="red", linestyle="--", label="95% CI")
ax.axhline(ci_high, color="red", linestyle="--")

ax.set_xlabel("Laboratory ID"); ax.set_ylabel("Measured value (x)")
ax.set_title("Interlaboratory Comparison - Outlier Analysis")
ax.legend()
st.pyplot(fig)

# ----------------------------
# 7. Sonuç Tablosu
# ----------------------------
st.subheader("📑 Analiz Sonuçları")
st.dataframe(results, use_container_width=True)

# ----------------------------
# 8. Recommendation
# ----------------------------
st.subheader("💡 Recommendation")
bad_labs = []
if "En_Status" in results:
    bad_labs += results.loc[results["En_Status"]=="UYUMSUZ","LabID"].astype(str).tolist()
if "MAD_Status" in results:
    bad_labs += results.loc[results["MAD_Status"]=="UYUMSUZ","LabID"].astype(str).tolist()

bad_labs = sorted(set(bad_labs))

if bad_labs:
    st.error(f"Uyumsuz laboratuvar(lar): {', '.join(bad_labs)}")
else:
    st.success("Tüm laboratuvarlar konsensüs ile uyumlu görünüyor.")
