import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ==============================
# SETTING HALAMAN
# ==============================

st.set_page_config(
    page_title="Prediksi Penyakit Paru-Paru",
    page_icon="🫁",
    layout="wide"
)


# ==============================
# STYLE TAMPILAN
# ==============================

st.markdown("""
<style>

body {
    background-color: #f4fbff;
}


.main-title {
    font-size: 40px;
    font-weight: bold;
    color: #0077b6;
}


.card {
    background-color: #0f4c75;
    padding: 25px;
    border-radius: 18px;
    color: white;
    height: 180px;
    box-shadow: 0px 5px 15px #bcdff5;
}


.card h2 {
    color: #90e0ef;
}


.result {
    font-size: 30px;
    font-weight: bold;
}


</style>
""", unsafe_allow_html=True)



# ==============================
# JUDUL
# ==============================

st.markdown(
"""
<div class="main-title">
🫁 Prediksi Risiko Penyakit Paru-Paru
</div>

<p>
Sistem deteksi awal menggunakan Machine Learning
dengan algoritma Random Forest.
</p>

""",
unsafe_allow_html=True
)



# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(
    "predic_tabel.csv",
    dtype=str
)



# ==============================
# DASHBOARD CARD
# ==============================

col1, col2, col3 = st.columns(3)


with col1:
    st.markdown("""
    <div class="card">
    <h2>① Input Form</h2>
    <p>
    Input 9 faktor risiko pasien
    seperti usia, merokok, dan aktivitas.
    </p>
    </div>
    """, unsafe_allow_html=True)


with col2:
    st.markdown("""
    <div class="card">
    <h2>② Prediction</h2>
    <p>
    Random Forest memberikan prediksi
    secara langsung.
    </p>
    </div>
    """, unsafe_allow_html=True)



with col3:
    st.markdown("""
    <div class="card">
    <h2>③ Probabilitas</h2>
    <p>
    Menampilkan nilai kemungkinan
    risiko penyakit.
    </p>
    </div>
    """, unsafe_allow_html=True)



st.write("")



# ==============================
# PREPROCESSING DATA
# ==============================


mapping = {

"Usia":
{
"Muda":0,
"Tua":1
},

"Jenis_Kelamin":
{
"Wanita":0,
"Pria":1
},

"Merokok":
{
"Pasif":0,
"Aktif":1
},

"Bekerja":
{
"Tidak":0,
"Ya":1
},

"Rumah_Tangga":
{
"Tidak":0,
"Ya":1
},

"Aktivitas_Begadang":
{
"Tidak":0,
"Ya":1
},

"Aktivitas_Olahraga":
{
"Jarang":0,
"Sering":1
},

"Asuransi":
{
"Tidak":0,
"Ada":1
},

"Penyakit_Bawaan":
{
"Tidak":0,
"Ada":1
},

"Hasil":
{
"Tidak":0,
"Ya":1
}

}



data = df.copy()



for kolom, nilai in mapping.items():

    data[kolom] = (
        data[kolom]
        .str.strip()
        .map(nilai)
    )



fitur = [
"Usia",
"Jenis_Kelamin",
"Merokok",
"Bekerja",
"Rumah_Tangga",
"Aktivitas_Begadang",
"Aktivitas_Olahraga",
"Asuransi",
"Penyakit_Bawaan"
]



X = data[fitur]

y = data["Hasil"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# ==============================
# MODEL
# ==============================

model = RandomForestClassifier(

    n_estimators=100,
    max_depth=10,
    random_state=42

)


model.fit(
    X_train,
    y_train
)



# ==============================
# SIDEBAR
# ==============================


menu = st.sidebar.selectbox(
    "Menu",
    [
        "Dashboard",
        "Prediksi Pasien",
        "Evaluasi Model"
    ]
)



# ==============================
# DASHBOARD
# ==============================


if menu == "Dashboard":

    st.subheader("📊 Informasi Model")

    a,b,c = st.columns(3)

    a.metric(
        "Jumlah Data",
        len(df)
    )

    b.metric(
        "Algoritma",
        "Random Forest"
    )

    c.metric(
        "Fitur",
        "9 Faktor Risiko"
    )



# ==============================
# PREDIKSI
# ==============================


elif menu == "Prediksi Pasien":


    st.subheader(
        "📝 Input Data Pasien"
    )


    col1,col2,col3 = st.columns(3)



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
            "Merokok",
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
            "Begadang",
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



    if st.button(
        "🔍 Mulai Prediksi",
        use_container_width=True
    ):


        input_data = np.array([[

        mapping["Usia"][usia],
        mapping["Jenis_Kelamin"][gender],
        mapping["Merokok"][rokok],
        mapping["Bekerja"][kerja],
        mapping["Rumah_Tangga"][rumah],
        mapping["Aktivitas_Begadang"][begadang],
        mapping["Aktivitas_Olahraga"][olahraga],
        mapping["Asuransi"][asuransi],
        mapping["Penyakit_Bawaan"][bawaan]

        ]])


        peluang = model.predict_proba(
            input_data
        )[0][1]


        if peluang >= 0.5:

            st.error(
            "🔴 Risiko Tinggi Penyakit Paru-Paru"
            )

        else:

            st.success(
            "🟢 Risiko Rendah Penyakit Paru-Paru"
            )


        st.write(
        f"Probabilitas Risiko: {peluang*100:.2f}%"
        )


        st.progress(
            float(peluang)
        )



# ==============================
# EVALUASI MODEL
# ==============================


else:

    st.subheader(
        "📈 Evaluasi Model"
    )


    prediksi = model.predict(X_test)


    col1,col2,col3,col4 = st.columns(4)


    col1.metric(
        "Accuracy",
        round(
            accuracy_score(y_test,prediksi),
            2
        )
    )


    col2.metric(
        "Precision",
        round(
            precision_score(y_test,prediksi),
            2
        )
    )


    col3.metric(
        "Recall",
        round(
            recall_score(y_test,prediksi),
            2
        )
    )


    col4.metric(
        "F1 Score",
        round(
            f1_score(y_test,prediksi),
            2
        )
    )



    st.subheader(
        "📊 Feature Importance"
    )


    hasil_fitur = pd.DataFrame({

        "Faktor":
        fitur,

        "Nilai":
        model.feature_importances_

    })


    st.bar_chart(
        hasil_fitur.set_index("Faktor")
    )


    st.subheader(
        "Confusion Matrix"
    )


    cm = confusion_matrix(
        y_test,
        prediksi
    )


    fig, ax = plt.subplots()

    ax.imshow(cm)

    ax.set_xlabel(
        "Prediksi"
    )

    ax.set_ylabel(
        "Aktual"
    )


    st.pyplot(fig)



st.caption(
"Machine Learning Project - Random Forest"
)