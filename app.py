import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =========================
# KONFIGURASI HALAMAN
# =========================

st.set_page_config(
    page_title="Prediksi Penyakit Paru-Paru",
    page_icon="🫁",
    layout="wide"
)


# =========================
# STYLE TAMPILAN
# =========================

# =========================
# STYLE MEDICAL UI
# =========================

st.markdown("""
<style>


/* Background */

.stApp{

background:
linear-gradient(
135deg,
#f8ffff,
#eaf7f7
);

}


/* Header */

.title{

font-size:38px;
font-weight:800;
color:#168aad;

}


.subtitle{

font-size:18px;
color:#607d8b;

}



/* Card */

.card{

background:#ffffff;

padding:25px;

border-radius:22px;

box-shadow:
0 8px 25px rgba(22,138,173,0.15);

border:1px solid #d9eeee;

}



/* Judul card */

.card-title{

font-size:22px;

font-weight:700;

color:#168aad;

}


/* angka */

.card h2{

color:#0f9d8a;

}



/* isi card */

.card p{

color:#607d8b;

font-size:16px;

}



/* Info box */

.stAlert{

background:#e7f8f5;

border-radius:18px;

color:#276749;

}



/* Sidebar */

section[data-testid="stSidebar"]{

background:

linear-gradient(
180deg,
#dff7f3,
#ffffff
);

}



section[data-testid="stSidebar"] *{

color:#276749 !important;

}



/* Select menu */

div[data-baseweb="select"] > div{

background:white;

border-radius:15px;

border:1px solid #b8e3df;

}



/* Button */

.stButton button{

background:#16a085;

color:white;

border-radius:25px;

font-weight:bold;

height:45px;

}



.stButton button:hover{

background:#087f5b;

color:white;

}



/* Metric */

[data-testid="metric-container"]{

background:white;

border-radius:18px;

padding:18px;

box-shadow:
0 6px 18px rgba(0,150,170,0.15);

}



</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.markdown(
"""
<div class="title">
Sistem Prediksi Penyakit Paru-Paru
</div>

<div class="subtitle">
Analisis risiko kesehatan menggunakan Machine Learning Random Forest
</div>

<br>

""",
unsafe_allow_html=True
)



# =========================
# SIDEBAR
# =========================


menu = st.sidebar.selectbox(
    "Menu Aplikasi",
    [
        "Dashboard",
        "Prediksi Pasien",
        "Analisis Model"
    ]
)



# =========================
# LOAD DATA
# =========================


df = pd.read_csv(
    "predic_tabel.csv",
    dtype=str
)



# =========================
# MAPPING DATA
# =========================


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



data=df.copy()


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



X=data[fitur]

y=data["Hasil"]



X_train,X_test,y_train,y_test = train_test_split(

X,
y,
test_size=0.2,
random_state=42

)



# =========================
# MODEL RANDOM FOREST
# =========================


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
# DASHBOARD
# =========================


if menu=="Dashboard":


    st.subheader("Dashboard Sistem")


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
        untuk pelatihan model.

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
        Machine Learning.

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

        Digunakan untuk analisis.

        </div>
        """,
        unsafe_allow_html=True
        )


    st.info(
        "Aplikasi ini digunakan untuk prediksi awal berdasarkan faktor risiko pasien."
    )



# =========================
# PREDIKSI
# =========================


elif menu=="Prediksi Pasien":


    st.subheader(
        "Input Data Pasien"
    )


    col1,col2,col3 = st.columns(3)



    with col1:

        usia=st.selectbox(
        "Usia",
        ["Muda","Tua"]
        )


        gender=st.selectbox(
        "Jenis Kelamin",
        ["Wanita","Pria"]
        )


        rokok=st.selectbox(
        "Merokok",
        ["Pasif","Aktif"]
        )



    with col2:

        kerja=st.selectbox(
        "Bekerja",
        ["Tidak","Ya"]
        )


        rumah=st.selectbox(
        "Rumah Tangga",
        ["Tidak","Ya"]
        )


        begadang=st.selectbox(
        "Begadang",
        ["Tidak","Ya"]
        )



    with col3:

        olahraga=st.selectbox(
        "Olahraga",
        ["Jarang","Sering"]
        )


        asuransi=st.selectbox(
        "Asuransi",
        ["Tidak","Ada"]
        )


        bawaan=st.selectbox(
        "Penyakit Bawaan",
        ["Tidak","Ada"]
        )



    if st.button(
        "Analisis Risiko"
    ):


        input_data=np.array([[


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


        risiko=model.predict_proba(
        input_data
        )[0][1]



        st.write(
        f"Tingkat Risiko: {risiko*100:.2f}%"
        )


        st.progress(
        float(risiko)
        )



        if risiko >=0.5:


            st.markdown(
            """
            <div class="risk-high">

            <h2>Risiko Tinggi</h2>

            Model mendeteksi kemungkinan risiko
            penyakit paru-paru.

            </div>

            """,
            unsafe_allow_html=True
            )



        else:


            st.markdown(
            """
            <div class="risk-low">

            <h2>Risiko Rendah</h2>

            Hasil menunjukkan risiko lebih rendah
            berdasarkan data pasien.

            </div>

            """,
            unsafe_allow_html=True
            )



# =========================
# ANALISIS MODEL
# =========================


else:


    st.subheader(
    "Evaluasi Model"
    )


    pred=model.predict(X_test)



    a,b,c,d=st.columns(4)


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



    st.subheader(
    "Faktor Risiko Utama"
    )


    importance=pd.DataFrame({

    "Faktor":fitur,

    "Nilai":model.feature_importances_

    })


    st.bar_chart(
    importance.set_index("Faktor")
    )



st.caption(
"Project Machine Learning - Random Forest"
)