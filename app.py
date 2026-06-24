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


# ======================
# CONFIG
# ======================

st.set_page_config(
    page_title="Prediksi Penyakit Paru-Paru",
    page_icon="🫁",
    layout="wide"
)


# ======================
# STYLE
# ======================

st.markdown("""
<style>


.stApp{
background:linear-gradient(
135deg,#f8ffff,#eaf8f6);
}


.title{
font-size:42px;
font-weight:800;
color:#0f7c7e;
}


.subtitle{
font-size:18px;
color:#64748b;
}



.card{

background:white;

padding:25px;

border-radius:22px;

box-shadow:
0px 8px 25px rgba(15,124,126,0.15);

border:1px solid #d8eeee;

}


.card-title{

font-size:22px;

font-weight:bold;

color:#0f7c7e;

}



p{

color:#475569 !important;

}


h1,h2,h3{

color:#0f7c7e !important;

}



/* sidebar */

section[data-testid="stSidebar"]{

background:#e8f8f5;

}


section[data-testid="stSidebar"] *{

color:#166534 !important;

}



/* select */

div[data-baseweb="select"] > div{

background:white !important;

border-radius:15px !important;

}


div[data-baseweb="select"] *{

color:#334155 !important;

}



/* button */

.stButton button{

background:#16a085;

color:white !important;

border-radius:25px;

font-weight:bold;

height:45px;

}



.stButton button:hover{

background:#087f5b;

}



/* risk */

.risk-low{

background:#ecfdf5;

padding:25px;

border-radius:20px;

border-left:8px solid #22c55e;

}



.risk-high{

background:#fff1f2;

padding:25px;

border-radius:20px;

border-left:8px solid #ef4444;

}


</style>

""",
unsafe_allow_html=True)



# ======================
# HEADER
# ======================


st.markdown(
"""
<div class="title">
Sistem Prediksi Penyakit Paru-Paru
</div>

<div class="subtitle">
Prediksi kesehatan menggunakan Machine Learning Random Forest
</div>

<br>
""",
unsafe_allow_html=True
)



# ======================
# MENU
# ======================

menu = st.sidebar.selectbox(
    "Menu",
    [
        "Dashboard",
        "Prediksi Pasien",
        "Analisis Model"
    ]
)



# ======================
# DATA
# ======================


df = pd.read_csv(
    "predic_tabel.csv",
    dtype=str
)



# ======================
# ENCODING
# ======================


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


for col,value in mapping.items():

    data[col]=(
        data[col]
        .str.strip()
        .map(value)
    )



fitur=[

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



# ======================
# TRAIN MODEL
# SAMA DENGAN ANALISIS
# ======================


X_train,X_test,y_train,y_test=train_test_split(

X,
y,

test_size=0.2,

random_state=42,

stratify=y

)



model=RandomForestClassifier(

n_estimators=100,

max_depth=10,

random_state=42,

n_jobs=-1

)


model.fit(
X_train,
y_train
)



# ======================
# DASHBOARD
# ======================


if menu=="Dashboard":


    st.subheader("Dashboard")


    a,b,c=st.columns(3)


    with a:

        st.markdown(
        """
        <div class="card">

        <div class="card-title">
        Dataset
        </div>

        <h2>30.000 Data</h2>

        Data pasien untuk training.

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

        Metode klasifikasi.

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



# ======================
# PREDIKSI
# ======================


elif menu=="Prediksi Pasien":


    st.subheader("Input Data Pasien")


    col1,col2,col3=st.columns(3)


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



    if st.button("Analisis Risiko"):


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


        hasil=model.predict(input_data)[0]


        probabilitas=model.predict_proba(input_data)[0]



        if hasil==1:


            st.markdown(
            f"""
            <div class="risk-high">

            <h2>Risiko Penyakit Paru-Paru</h2>

            Probabilitas Risiko:
            {probabilitas[1]*100:.2f}%


            </div>
            """,
            unsafe_allow_html=True
            )


        else:


            st.markdown(
            f"""
            <div class="risk-low">

            <h2>Risiko Rendah</h2>

            Probabilitas Risiko:
            {probabilitas[1]*100:.2f}%


            </div>
            """,
            unsafe_allow_html=True
            )



# ======================
# ANALISIS MODEL
# ======================


else:


    st.subheader("Evaluasi Model")


    pred=model.predict(X_test)



    a,b,c,d=st.columns(4)


    a.metric(
    "Accuracy",
    round(
    accuracy_score(y_test,pred),
    3)
    )


    b.metric(
    "Precision",
    round(
    precision_score(y_test,pred),
    3)
    )


    c.metric(
    "Recall",
    round(
    recall_score(y_test,pred),
    3)
    )


    d.metric(
    "F1 Score",
    round(
    f1_score(y_test,pred),
    3)
    )


    st.subheader(
    "Feature Importance"
    )


    importance=pd.DataFrame({

    "Fitur":fitur,

    "Nilai":model.feature_importances_

    })


    st.bar_chart(
    importance.set_index("Fitur")
    )



st.caption(
"Project Machine Learning - Random Forest"
)