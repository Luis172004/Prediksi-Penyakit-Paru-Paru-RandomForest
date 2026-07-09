import streamlit as st
import pandas as pd
import joblib


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
""", unsafe_allow_html=True)


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
# LOAD MODEL & ENCODER
# (dilatih sekali di notebook, bukan retrain tiap
#  aplikasi dibuka -> hasil konsisten dgn laporan)
# ======================

@st.cache_resource
def load_artifacts():
    model = joblib.load("model_rf_v2.pkl")
    le_dict = joblib.load("label_encoders_v2.pkl")
    return model, le_dict

model, le_dict = load_artifacts()

# SINKRONISASI: Menggunakan spasi agar cocok dengan output train_v2.py
fitur = [
    "Usia",
    "Jenis_Kelamin",
    "Merokok",
    "Bekerja",
    "Rumah_Tangga",
    "Aktivitas Begadang",
    "Aktivitas Olahraga",
    "Asuransi",
    "Penyakit_Bawaan"
]

# Opsi dropdown diambil langsung dari kelas yang dikenal
# LabelEncoder saat training -> dijamin selalu sinkron,
# tidak bisa "typo" beda dengan yang dipelajari model.
opsi = {col: list(le_dict[col].classes_) for col in fitur}


# ======================
# DASHBOARD
# ======================

if menu == "Dashboard":

    st.subheader("Dashboard")

    a, b, c = st.columns(3)

    with a:
        st.markdown(
        """
        <div class="card">
        <div class="card-title">Dataset</div>
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
        <div class="card-title">Algoritma</div>
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
        <div class="card-title">Faktor Risiko</div>
        <h2>9 Faktor</h2>
        Digunakan untuk prediksi.
        </div>
        """,
        unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "⚠️ **Catatan:** pada dataset ini, pola beberapa fitur (mis. Merokok, "
        "Aktivitas Olahraga) tidak selalu searah dengan intuisi medis umum. "
        "Model memprediksi murni berdasarkan pola statistik data latih, "
        "bukan berdasarkan aturan medis manual."
    )


# ======================
# PREDIKSI
# ======================

elif menu == "Prediksi Pasien":

    st.subheader("Input Data Pasien")

    col1, col2, col3 = st.columns(3)

    with col1:
        usia = st.selectbox("Usia", opsi["Usia"])
        gender = st.selectbox("Jenis Kelamin", opsi["Jenis_Kelamin"])
        rokok = st.selectbox("Merokok", opsi["Merokok"])

    with col2:
        kerja = st.selectbox("Bekerja", opsi["Bekerja"])
        rumah = st.selectbox("Rumah Tangga", opsi["Rumah_Tangga"])
        begadang = st.selectbox("Begadang", opsi["Aktivitas Begadang"])

    with col3:
        olahraga = st.selectbox("Olahraga", opsi["Aktivitas Olahraga"])
        asuransi = st.selectbox("Asuransi", opsi["Asuransi"])
        bawaan = st.selectbox("Penyakit Bawaan", opsi["Penyakit_Bawaan"])

    # SINKRONISASI: Kunci dicocokkan dengan nama fitur ber-spasi
    input_values = {
        "Usia": usia,
        "Jenis_Kelamin": gender,
        "Merokok": rokok,
        "Bekerja": kerja,
        "Rumah_Tangga": rumah,
        "Aktivitas Begadang": begadang,
        "Aktivitas Olahraga": olahraga,
        "Asuransi": asuransi,
        "Penyakit_Bawaan": bawaan
    }

    if st.button("Analisis Risiko"):

        # ===========================
        # Encode input persis dengan
        # LabelEncoder hasil training
        # ===========================
        input_df = pd.DataFrame([input_values])[fitur]
        for col in fitur:
            input_df[col] = le_dict[col].transform(input_df[col])

        # ===========================
        # Prediksi murni dari model
        # (tidak ada penyesuaian manual
        #  yang bisa bertentangan dgn model)
        # ===========================
        pred = model.predict(input_df)[0]
        prob_positif = model.predict_proba(input_df)[0][1] * 100

        if pred == 1:
            st.markdown(
                f"""
                <div class="risk-high">
                <h2>⚠️ Prediksi: POSITIF Penyakit Paru-Paru</h2>
                <hr>
                <b>Probabilitas Positif Model :</b>
                <span style="font-size:28px;color:red;">
                {prob_positif:.2f}%
                </span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="risk-low">
                <h2>✅ Prediksi: NEGATIF Penyakit Paru-Paru</h2>
                <hr>
                <b>Probabilitas Positif Model :</b>
                <span style="font-size:28px;color:green;">
                {prob_positif:.2f}%
                </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.caption(
            "Prediksi ini murni output model Random Forest yang dilatih pada "
            "predic_tabel.csv (akurasi ±94,33% pada data uji), tanpa "
            "penyesuaian skor manual."
        )


# ======================
# ANALISIS MODEL
# ======================

elif menu == "Analisis Model":

    st.subheader("Analisis Model")

    importances = pd.DataFrame({
        "Fitur": fitur,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.markdown("#### Feature Importance")
    st.bar_chart(importances.set_index("Fitur"))

    st.markdown("#### Tabel Feature Importance")
    st.dataframe(
        importances.reset_index(drop=True),
        use_container_width=True
    )

    st.caption(
        "Prediksi ini murni output model Random Forest yang dilatih pada "
        "predic_tabel.csv (akurasi ±94,33% pada data uji), tanpa "
        "penyesuaian skor manual."
    )