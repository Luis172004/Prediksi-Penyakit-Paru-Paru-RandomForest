import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Membaca dataset baru
df = pd.read_csv('predic_tabel.csv')

# --- SINKRONISASI NAMA KOLOM SESUAI APP.PY ANDA ---
df = df.rename(columns={
    'Aktivitas_Begadang': 'Aktivitas Begadang',
    'Aktivitas_Olahraga': 'Aktivitas Olahraga'
})

# 2. Pisahkan Fitur dan Target
X = df.drop(columns=['No', 'Hasil'])
y = df['Hasil']

# 3. Encoding Data Kategorikal
label_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode Target (Hasil)
le_y = LabelEncoder()
y = le_y.fit_transform(y)
label_encoders['Hasil'] = le_y

# 4. Latih Model Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X, y)

# 5. Simpan file .pkl BARU yang dijamin COCOK dengan app.py Anda
joblib.dump(model_rf, 'model_rf.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("✨ SUKSES! File model_rf.pkl dan label_encoders.pkl baru yang SINKRON telah dibuat!")
