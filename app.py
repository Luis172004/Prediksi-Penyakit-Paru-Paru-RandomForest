import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


st.title("Prediksi Penyakit Paru-Paru")
st.write("Menggunakan Algoritma Random Forest")


# Load dataset
df = pd.read_csv("predic_tabel.csv", dtype=str)


# Encoding
mapping = {
    'Usia': {'Muda':0, 'Tua':1},
    'Jenis_Kelamin': {'Wanita':0, 'Pria':1},
    'Merokok': {'Pasif':0, 'Aktif':1},
    'Bekerja': {'Tidak':0, 'Ya':1},
    'Rumah_Tangga': {'Tidak':0, 'Ya':1},
    'Aktivitas_Begadang': {'Tidak':0, 'Ya':1},
    'Aktivitas_Olahraga': {'Jarang':0, 'Sering':1},
    'Asuransi': {'Tidak':0, 'Ada':1},
    'Penyakit_Bawaan': {'Tidak':0, 'Ada':1},
    'Hasil': {'Tidak':0,'Ya':1}
}


data = df.copy()

for col, val in mapping.items():
    data[col] = data[col].str.strip().map(val)


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
    X,y,test_size=0.2,random_state=42
)


model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)


model.fit(X_train,y_train)


st.subheader("Input Data Pasien")


usia = st.selectbox(
    "Usia",
    ["Muda","Tua"]
)

gender = st.selectbox(
    "Jenis Kelamin",
    ["Wanita","Pria"]
)

rokok = st.selectbox(
    "Merokok",
    ["Pasif","Aktif"]
)

kerja = st.selectbox(
    "Bekerja",
    ["Tidak","Ya"]
)

rumah = st.selectbox(
    "Rumah Tangga",
    ["Tidak","Ya"]
)

begadang = st.selectbox(
    "Aktivitas Begadang",
    ["Tidak","Ya"]
)

olahraga = st.selectbox(
    "Aktivitas Olahraga",
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


if st.button("🔍 Prediksi"):


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

    prob = model.predict_proba(input_data)[0]


    if hasil == 1:
        st.error(
        " Prediksi: Berisiko Penyakit Paru-Paru"
        )
    else:
        st.success(
        "✅ Prediksi: Tidak Berisiko"
        )


    st.write(
    f"Probabilitas Risiko: {prob[1]*100:.2f}%"
    )


st.subheader("Feature Importance")


importance = pd.DataFrame({
    "Fitur": fitur,
    "Nilai": model.feature_importances_
})


st.bar_chart(
    importance.set_index("Fitur")
)