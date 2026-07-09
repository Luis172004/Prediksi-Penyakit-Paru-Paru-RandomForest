import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv('predic_tabel.csv')
df = df.rename(columns={'Aktivitas_Begadang': 'Aktivitas Begadang', 'Aktivitas_Olahraga': 'Aktivitas Olahraga'})
X = df.drop(columns=['No', 'Hasil'])
y = df['Hasil']

le_dict = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

le_y = LabelEncoder()
y = le_y.fit_transform(y)
le_dict['Hasil'] = le_y

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X, y)

# Kita ganti namanya menjadi v2 agar Git dipaksa membaca file baru!
joblib.dump(model_rf, 'model_rf_v2.pkl')
joblib.dump(le_dict, 'label_encoders_v2.pkl')
print("✨ SUKSES! File model_rf_v2.pkl dan label_encoders_v2.pkl berhasil dibuat!")
