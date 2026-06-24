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


# ==========================
# KONFIGURASI HALAMAN
# ==========================

st.set_page_config(
    page_title="LungCare AI",
    page_icon="🫁",
    layout="wide"
)


# ==========================
# STYLE
# ==========================

st.markdown("""
<style>

.stApp{
    background-color:#f5f9fc;
}


.title{
    font-size:42px;
    font-weight:700;
    color:#075985;
}


.subtitle{
    color:#555;
    font-size:18px;
}


.card{

background:white;
padding:25px;
border-radius:15px;
box-shadow:0px 4px 15px #d9e8f5;

}


.card-title{

font-size:22px;
font-weight:bold;
color:#075985;

}


.result-high{

background:#ffe5e5;
padding:20px;
border-radius:15px;
color:#b91c1c;

}


.result-low{

background:#dcfce7;
padding:20px;
border-radius:15px;
color:#166534;

}

</style>

""", unsafe_allow_html=True)



# ==========================
# HEADER
# ==========================

st.markdown(
"""
<div class="title">
LungCare AI
</div>

<div class="subtitle">
Sistem Prediksi Risiko Penyakit Paru-Paru Menggunakan Random Forest
</div>

<br>
""",
unsafe_allow_html=True
)



# ==========================
# SIDEBAR
# ==========================

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Dashboard",
        "Prediksi Pasien",
        "Analisis Model"
    ]
)



# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(
    "predic_tabel.csv",
    dtype=str
)



# ==========================
# DATA PREPROCESSING
# ==========================

mapping = {

"Usia":{
"Muda":0,
"Tua":1
},

"Jenis_Kelamin":{
"Wanita":0,
"Pria":1
},

"Merokok":{
"Pasif":0,
"Aktif":1
},

"Bekerja":{
"Tidak":0,
"Ya":1
},

"Rumah_Tangga":{
"Tidak":0,
"Ya":1
},

"Aktivitas_Begadang":{
"Tidak":0,
"Ya":1
},

"Aktivitas_Olahraga":{
"Jarang":0,
"Sering":1
},

"Asuransi":{
"Tidak":0,
"Ada":1
},

"Penyakit_Bawaan":{
"Tidak":0,
"Ada":1
},

"Hasil":{
"Tidak":0,
"Ya":1
}

}


data = df.copy()


for col, value in mapping.items():

    data[col] = (
        data[col]
        .str.strip()
        .map(value)
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



# ==========================
# MODEL
# ==========================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)


model.fit(
    X_train,
    y_train
)



# ==========================
# DASHBOARD
# ==========================

if menu == "Dashboard":


    st.subheader("Dashboard")


    a,b,c = st.columns(3)


    with a:
        st.markdown(
        """
        <div class="card">

        <div class="card-title">
        Dataset
        </div>

        <h2>30.000+</h2>

        Data pasien digunakan
        untuk training model.

        </div>
        """,
        unsafe_allow_html=True
        )


    with b:
        st.markdown(
        """
        <div class="card">

        <div class="card-title">
        Algoritma
        </div>

        <h2>Random Forest</h2>

        Metode klasifikasi
        machine learning.

        </div>
        """,
        unsafe_allow_html=True
        )



    with c:
        st.markdown(
        """
        <div class="card">

        <div class="card-title">
        Faktor Risiko
        </div>

        <h2>9 Faktor</h2>

        Digunakan untuk prediksi.

        </div>
        """,
        unsafe_allow_html=True
        )


    st.write("")

    st.info(
        "LungCare AI membantu melakukan prediksi awal berdasarkan faktor risiko pasien. "
        "Hasil prediksi bukan pengganti diagnosis dokter."
    )



# ==========================
# PREDIKSI
# ==========================

elif menu == "Prediksi Pasien":


    st.subheader("Prediksi Risiko Pasien")


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
        "Analisis Risiko",
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


        probability = model.predict_proba(
            input_data
        )[0][1]


        st.write(
        f"Tingkat Risiko: {probability*100:.2f}%"
        )


        st.progress(
            float(probability)
        )


        if probability >= 0.5:

            st.markdown(
            """
            <div class="result-high">

            <h2>Risiko Tinggi</h2>

            Hasil prediksi menunjukkan adanya kemungkinan
            risiko penyakit paru-paru.

            </div>
            """,
            unsafe_allow_html=True
            )


        else:

            st.markdown(
            """
            <div class="result-low">

            <h2>Risiko Rendah</h2>

            Hasil prediksi menunjukkan risiko lebih rendah
            berdasarkan data yang dimasukkan.

            </div>
            """,
            unsafe_allow_html=True
            )



# ==========================
# ANALISIS MODEL
# ==========================

else:


    st.subheader("Evaluasi Model")


    pred = model.predict(X_test)


    a,b,c,d = st.columns(4)


    a.metric(
        "Accuracy",
        round(
            accuracy_score(y_test,pred),
            3
        )
    )


    b.metric(
        "Precision",
        round(
            precision_score(y_test,pred),
            3
        )
    )


    c.metric(
        "Recall",
        round(
            recall_score(y_test,pred),
            3
        )
    )


    d.metric(
        "F1 Score",
        round(
            f1_score(y_test,pred),
            3
        )
    )


    st.subheader("Faktor Risiko")


    importance = pd.DataFrame({

    "Faktor":fitur,

    "Nilai":model.feature_importances_

    })


    importance = importance.sort_values(
        "Nilai",
        ascending=False
    )


    st.bar_chart(
        importance.set_index("Faktor")
    )


st.caption(
"LungCare AI - Machine Learning Project"
)