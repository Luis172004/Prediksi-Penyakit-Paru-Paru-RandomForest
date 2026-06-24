import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Penyakit Paru-Paru",
    page_icon="🫁",
    layout="wide"
)

# ─────────────────────────────────────────────
# CSS STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Background utama */
    .stApp {
        background-color: #f0f4f8;
    }

    /* Header besar */
    .main-header {
        background: linear-gradient(135deg, #1a6fa3 0%, #0d4f7c 100%);
        padding: 32px 40px;
        border-radius: 16px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 4px 16px rgba(13, 79, 124, 0.25);
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 8px 0;
    }
    .main-header p {
        font-size: 1rem;
        margin: 0;
        opacity: 0.88;
    }

    /* Card ringkasan dashboard */
    .card {
        background: white;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        border-left: 5px solid #1a6fa3;
        margin-bottom: 16px;
        height: 100%;
    }
    .card h4 {
        color: #1a6fa3;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0 0 8px 0;
    }
    .card p {
        color: #374151;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.5;
    }
    .card-green {
        border-left-color: #16a34a;
    }
    .card-green h4 { color: #16a34a; }

    /* Hasil prediksi */
    .result-box {
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 20px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    .result-high {
        background: #fef2f2;
        border: 2px solid #ef4444;
        color: #b91c1c;
    }
    .result-low {
        background: #f0fdf4;
        border: 2px solid #22c55e;
        color: #15803d;
    }

    /* Metric card */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a6fa3;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 4px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d4f7c;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: white !important;
        font-size: 0.95rem;
    }

    /* Tombol prediksi */
    .stButton > button {
        background: linear-gradient(135deg, #1a6fa3, #0d4f7c);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD & PREPROCESSING DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("predic_tabel.csv")

    # Mapping nilai kategorikal ke numerik
    mapping = {
        "Muda": 0, "Tua": 1,
        "Pria": 1, "Wanita": 0,
        "Ya": 1, "Tidak": 0,
        "Aktif": 1, "Tidak Aktif": 0,
        "Sering": 1, "Jarang": 0,
        "Ada": 1, "Tidak Ada": 0,
    }

    df_encoded = df.copy()
    for col in df_encoded.columns:
        df_encoded[col] = df_encoded[col].map(lambda x: mapping.get(x, x))

    # Pisahkan fitur dan target
    X = df_encoded.drop(columns=["Hasil"])
    y = df_encoded["Hasil"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Buat dan latih model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Hitung metrik
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
    }
    cm = confusion_matrix(y_test, y_pred)

    return model, X.columns.tolist(), metrics, cm, df

model, feature_names, metrics, cm, df = load_and_train()


# ─────────────────────────────────────────────
# SIDEBAR NAVIGASI
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫁 Menu Navigasi")
    st.markdown("---")
    halaman = st.radio(
        "",
        ["🏠 Dashboard", "🔍 Prediksi Pasien", "📊 Evaluasi Model"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#90caf9'>Universitas Nusa Putra<br>Teknik Informatika</small>",
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════
# HALAMAN: DASHBOARD
# ═══════════════════════════════════════════════
if halaman == "🏠 Dashboard":

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🫁 Prediksi Risiko Penyakit Paru-Paru</h1>
        <p>Sistem deteksi awal berbasis Machine Learning menggunakan Random Forest</p>
    </div>
    """, unsafe_allow_html=True)

    # 6 Dashboard Card
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h4>📋 Input Form</h4>
            <p>Form untuk memasukkan 9 faktor risiko pasien secara interaktif.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h4>⚡ Real-Time Prediction</h4>
            <p>Model Random Forest melakukan prediksi secara langsung saat data dimasukkan.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <h4>📈 Probabilitas Risiko</h4>
            <p>Menampilkan persentase kemungkinan risiko penyakit paru-paru pasien.</p>
        </div>
        """, unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
        <div class="card card-green">
            <h4>🔑 Feature Importance</h4>
            <p>Menampilkan faktor risiko yang paling berpengaruh terhadap hasil prediksi.</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div class="card card-green">
            <h4>📐 Model Metrics</h4>
            <p>Menampilkan Accuracy, Precision, Recall, dan F1 Score dari model.</p>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
        <div class="card card-green">
            <h4>🌐 Open Access</h4>
            <p>Aplikasi dapat diakses dan digunakan melalui Streamlit Cloud secara gratis.</p>
        </div>
        """, unsafe_allow_html=True)

    # Info dataset
    st.markdown("---")
    st.markdown("### 📁 Informasi Dataset")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{len(df)}</div>
            <div class="label">Total Data</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{len(feature_names)}</div>
            <div class="label">Jumlah Fitur</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{round(metrics['accuracy']*100, 1)}%</div>
            <div class="label">Akurasi Model</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗂️ Contoh Data")
    st.dataframe(df.head(10), use_container_width=True)


# ═══════════════════════════════════════════════
# HALAMAN: PREDIKSI PASIEN
# ═══════════════════════════════════════════════
elif halaman == "🔍 Prediksi Pasien":

    st.markdown("""
    <div class="main-header">
        <h1>🔍 Prediksi Risiko Pasien</h1>
        <p>Masukkan data pasien untuk memprediksi risiko penyakit paru-paru</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("form_prediksi"):
        st.markdown("#### 📋 Data Pasien")

        col1, col2, col3 = st.columns(3)

        with col1:
            usia = st.selectbox("🎂 Usia", ["Muda", "Tua"])
            jenis_kelamin = st.selectbox("⚧ Jenis Kelamin", ["Pria", "Wanita"])
            merokok = st.selectbox("🚬 Merokok", ["Ya", "Tidak"])

        with col2:
            bekerja = st.selectbox("💼 Bekerja", ["Ya", "Tidak"])
            rumah_tangga = st.selectbox("🏠 Rumah Tangga", ["Ya", "Tidak"])
            begadang = st.selectbox("🌙 Aktivitas Begadang", ["Sering", "Jarang"])

        with col3:
            olahraga = st.selectbox("🏃 Aktivitas Olahraga", ["Aktif", "Tidak Aktif"])
            asuransi = st.selectbox("🛡️ Asuransi", ["Ada", "Tidak Ada"])
            penyakit_bawaan = st.selectbox("🧬 Penyakit Bawaan", ["Ada", "Tidak Ada"])

        tombol = st.form_submit_button("🔍 Mulai Prediksi")

    if tombol:
        # Mapping input ke angka
        mapping = {
            "Muda": 0, "Tua": 1,
            "Pria": 1, "Wanita": 0,
            "Ya": 1, "Tidak": 0,
            "Aktif": 1, "Tidak Aktif": 0,
            "Sering": 1, "Jarang": 0,
            "Ada": 1, "Tidak Ada": 0,
        }

        input_data = np.array([[
            mapping[usia],
            mapping[jenis_kelamin],
            mapping[merokok],
            mapping[bekerja],
            mapping[rumah_tangga],
            mapping[begadang],
            mapping[olahraga],
            mapping[asuransi],
            mapping[penyakit_bawaan],
        ]])

        # Prediksi
        proba = model.predict_proba(input_data)[0]
        # Ambil probabilitas kelas positif (risiko tinggi = 1)
        # Cek urutan kelas
        kelas = model.classes_
        if 1 in kelas:
            idx_positif = list(kelas).index(1)
        else:
            idx_positif = 1
        risiko_persen = proba[idx_positif] * 100

        # Tampilkan hasil
        st.markdown("---")
        st.markdown("#### 🎯 Hasil Prediksi")

        if risiko_persen >= 50:
            st.markdown(f"""
            <div class="result-box result-high">
                🔴 Risiko Tinggi Penyakit Paru-Paru
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-low">
                🟢 Risiko Rendah Penyakit Paru-Paru
            </div>
            """, unsafe_allow_html=True)

        # Probabilitas dan progress bar
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.markdown(f"""
            <div class="metric-card" style="margin-top:8px">
                <div class="value">{risiko_persen:.1f}%</div>
                <div class="label">Probabilitas Risiko</div>
            </div>
            """, unsafe_allow_html=True)
        with col_r2:
            st.markdown("**Progress Risiko:**")
            st.progress(risiko_persen / 100)
            if risiko_persen >= 50:
                st.warning("⚠️ Disarankan untuk segera memeriksakan diri ke dokter spesialis paru.")
            else:
                st.success("✅ Tetap jaga kesehatan dan lakukan pemeriksaan rutin.")

        # Ringkasan input
        st.markdown("---")
        st.markdown("#### 📋 Ringkasan Input Pasien")
        ringkasan = {
            "Faktor": ["Usia", "Jenis Kelamin", "Merokok", "Bekerja", "Rumah Tangga",
                       "Aktivitas Begadang", "Aktivitas Olahraga", "Asuransi", "Penyakit Bawaan"],
            "Nilai": [usia, jenis_kelamin, merokok, bekerja, rumah_tangga,
                      begadang, olahraga, asuransi, penyakit_bawaan]
        }
        st.dataframe(pd.DataFrame(ringkasan), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# HALAMAN: EVALUASI MODEL
# ═══════════════════════════════════════════════
elif halaman == "📊 Evaluasi Model":

    st.markdown("""
    <div class="main-header">
        <h1>📊 Evaluasi Model Random Forest</h1>
        <p>Performa model berdasarkan data uji (test set)</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrik utama
    st.markdown("### 🏆 Metrik Performa")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{metrics['accuracy']*100:.1f}%</div>
            <div class="label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{metrics['precision']*100:.1f}%</div>
            <div class="label">Precision</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{metrics['recall']*100:.1f}%</div>
            <div class="label">Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{metrics['f1']*100:.1f}%</div>
            <div class="label">F1 Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_cm, col_fi = st.columns(2)

    # Confusion Matrix
    with col_cm:
        st.markdown("### 🔲 Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax_cm)
        classes = ["Negatif", "Positif"]
        tick_marks = np.arange(len(classes))
        ax_cm.set_xticks(tick_marks)
        ax_cm.set_xticklabels(classes)
        ax_cm.set_yticks(tick_marks)
        ax_cm.set_yticklabels(classes)
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")
        ax_cm.set_title("Confusion Matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, str(cm[i, j]),
                           ha="center", va="center",
                           color="white" if cm[i, j] > cm.max() / 2 else "black",
                           fontsize=14, fontweight="bold")
        fig_cm.tight_layout()
        st.pyplot(fig_cm)

    # Feature Importance
    with col_fi:
        st.markdown("### 🔑 Feature Importance")
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importances = importances[sorted_idx]

        fig_fi, ax_fi = plt.subplots(figsize=(5, 4))
        colors = ["#1a6fa3" if i == 0 else "#64b5f6" for i in range(len(sorted_features))]
        bars = ax_fi.barh(sorted_features[::-1], sorted_importances[::-1], color=colors[::-1])
        ax_fi.set_xlabel("Importance Score")
        ax_fi.set_title("Feature Importance")
        for bar, val in zip(bars, sorted_importances[::-1]):
            ax_fi.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                       f"{val:.3f}", va="center", fontsize=9)
        fig_fi.tight_layout()
        st.pyplot(fig_fi)

    # Tabel ringkasan
    st.markdown("---")
    st.markdown("### 📋 Tabel Feature Importance")
    fi_df = pd.DataFrame({
        "Fitur": sorted_features,
        "Importance Score": [round(v, 4) for v in sorted_importances],
        "Persentase (%)": [round(v * 100, 2) for v in sorted_importances]
    })
    st.dataframe(fi_df, use_container_width=True, hide_index=True)