import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# =========================
# KONFIGURASI HALAMAN
# =========================

st.set_page_config(
    page_title="Deteksi Penyakit Paru-Paru",
    page_icon="🫁",
    layout="wide"
)


# =========================
# STYLE CSS
# =========================

st.markdown("""
<style>

body {
    background-color: #f5fbff;
}

.main-title {
    font-size: 40px;
    font-weight: bold;
    color: #006d77;
}

.subtitle {
    font-size: 18px;
    color: #555;
}


.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px #d9eaf7;
}


.result {
    font-size: 28px;
    font-weight: bold;
}

</style>

""", unsafe_allow_html=True)



# =========================
# HEADER
# =========================


st.markdown(
"""
<div class="card">

<div class="main-title">
🫁 Sistem Prediksi Penyakit Paru-Paru
</div>

<p class="subtitle">
Deteksi awal risiko penyakit paru-paru menggunakan Machine Learning
dengan algoritma Random Forest.
</p>

</div>
""",
unsafe_allow_html=True
)



st.write("")



# =========================
# LOAD DATA
# =========================


df = pd.read_csv(
    "predic_tabel.csv",
    dtype=str
)



mapping = {

    'Usia':
    {'Muda':0,'Tua':1},

    'Jenis_Kelamin':
    {'Wanita':0,'Pria':1},

    'Merokok':
    {'Pasif':0,'Aktif':1},

    'Bekerja':
    {'Tidak':0,'Ya':1},

    'Rumah_Tangga':
    {'Tidak':0,'Ya':1},

    'Aktivitas_Begadang':
    {'Tidak':0,'Ya':1},

    'Aktivitas_Olahraga':
    {'Jarang':0,'Sering':1},

    'Asuransi':
    {'Tidak':0,'Ada':1},

    'Penyakit_Bawaan':
    {'Tidak':0,'Ada':1},

    'Hasil':
    {'Tidak':0,'Ya':1}

}


data = df.copy()


for kolom, nilai in mapping.items():

    data[kolom] = (
        data[kolom]
        .str.strip()
        .map(nilai)
    )



fitur = [

'Usia',
'Jenis_Kelamin',
'Merokok',
'Bekerja',
'Rumah_Tangga',
'Aktivitas_Begadang',
'Aktivitas_Olahraga',
'Asuransi',
'Penyakit_Bawaan'

]



X = data[fitur]

y = data["Hasil"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)


model.fit(
    X_train,
    y_train
)



# =========================
# SIDEBAR
# =========================


with st.sidebar:

    st.header("🫁 Informasi")

    st.write(
"""
Aplikasi ini membantu melakukan
prediksi awal berdasarkan faktor risiko.

Algoritma:
**Random Forest**

Data:
30.000 pasien
"""
)

    st.info(
        "Hasil prediksi bukan pengganti pemeriksaan dokter."
    )



# =========================
# INPUT PASIEN
# =========================


st.subheader("📋 Data Pasien")


col1, col2, col3 = st.columns(3)


with col1:

    usia = st.selectbox(
        "Usia",
        ["Muda","Tua"]
    )


    gender = st.selectbox(
        "Jenis Kelamin",
        ["Wanita","Pria"]
    )


    rokok = st.selectbox(
        "Status Merokok",
        ["Pasif","Aktif"]
    )



with col2:


    kerja = st.selectbox(
        "Bekerja",
        ["Tidak","Ya"]
    )


    rumah = st.selectbox(
        "Rumah Tangga",
        ["Tidak","Ya"]
    )


    begadang = st.selectbox(
        "Sering Begadang",
        ["Tidak","Ya"]
    )



with col3:


    olahraga = st.selectbox(
        "Olahraga",
        ["Jarang","Sering"]
    )


    asuransi = st.selectbox(
        "Asuransi",
        ["Tidak","Ada"]
    )


    bawaan = st.selectbox(
        "Penyakit Bawaan",
        ["Tidak","Ada"]
    )



st.write("")


# =========================
# PREDIKSI
# =========================


if st.button(
    "🔍 Mulai Prediksi",
    use_container_width=True
):


    input_data = np.array([[
        mapping['Usia'][usia],
        mapping['Jenis_Kelamin'][gender],
        mapping['Merokok'][rokok],
        mapping['Bekerja'][kerja],
        mapping['Rumah_Tangga'][rumah],
        mapping['Aktivitas_Begadang'][begadang],
        mapping['Aktivitas_Olahraga'][olahraga],
        mapping['Asuransi'][asuransi],
        mapping['Penyakit_Bawaan'][bawaan]

    ]])


    hasil = model.predict(input_data)[0]

    peluang = model.predict_proba(input_data)[0][1]



    st.write("")


    if hasil == 1:

        st.error(
        "⚠️ Pasien memiliki risiko penyakit paru-paru"
        )


    else:

        st.success(
        "✅ Pasien memiliki risiko rendah"
        )



    st.metric(
        "Kemungkinan Risiko",
        f"{peluang*100:.2f}%"
    )



# =========================
# FEATURE IMPORTANCE
# =========================


st.subheader("📊 Faktor Risiko Utama")


importance = pd.DataFrame({

    "Faktor":
    fitur,

    "Nilai":
    model.feature_importances_

})


importance = importance.sort_values(
    "Nilai",
    ascending=False
)



st.bar_chart(
    importance.set_index("Faktor")
)



st.caption(
"Developed for Machine Learning Project - Random Forest"
)