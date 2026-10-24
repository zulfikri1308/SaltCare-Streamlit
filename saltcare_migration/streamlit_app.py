from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_best.joblib"
DB_DIR = BASE_DIR / "instance"
DB_PATH = DB_DIR / "saltcare_streamlit.db"

st.set_page_config(page_title="SaltCare | Prediksi Hipertensi", page_icon="💙", layout="wide")

CATEGORY_MAPPINGS = {
    "BP_History": {"Normal": 0, "Prehypertension": 1, "Hypertension": 2},
    "Exercise_Level": {"Low": 0, "Moderate": 1, "High": 2},
    "Family_History": {"No": 0, "Yes": 1},
    "Smoking_Status": {"Non-Smoker": 0, "Smoker": 1},
    "Medication": {"ACE Inhibitor": 0, "Beta Blocker": 1, "Diuretic": 2, "None": 3, "Other": 4},
}
TARGET_LABELS = {0: "Tidak terklasifikasi hipertensi", 1: "Terklasifikasi hipertensi"}
FEATURE_LABELS = {
    "Age": "Usia", "Salt_Intake": "Konsumsi garam", "Stress_Score": "Skor stres",
    "BP_History": "Riwayat tekanan darah", "Sleep_Duration": "Durasi tidur",
    "BMI": "Indeks massa tubuh (BMI)", "Medication": "Penggunaan obat",
    "Family_History": "Riwayat hipertensi keluarga", "Exercise_Level": "Tingkat aktivitas olahraga",
    "Smoking_Status": "Status merokok",
}

@st.cache_resource(show_spinner="Memuat model Random Forest...")
def load_artifact():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or "model" not in artifact or "feature_names" not in artifact:
        raise ValueError("Format artefak model tidak sesuai.")
    return artifact


def encode_input(input_data, feature_names):
    encoded = dict(input_data)
    for feature, mapping in CATEGORY_MAPPINGS.items():
        if feature in encoded:
            if encoded[feature] not in mapping:
                raise ValueError(f"Kategori {encoded[feature]!r} tidak valid untuk {feature}.")
            encoded[feature] = mapping[encoded[feature]]
    missing = [f for f in feature_names if f not in encoded]
    if missing:
        raise ValueError(f"Fitur belum tersedia: {missing}")
    return pd.DataFrame([[encoded[f] for f in feature_names]], columns=feature_names), encoded


def init_db():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH, timeout=10) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            input_data TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            prediction_label TEXT NOT NULL,
            probabilities TEXT,
            best_split TEXT,
            best_accuracy REAL
        )""")


def save_history(input_data, result, artifact):
    init_db()
    with sqlite3.connect(DB_PATH, timeout=10) as con:
        con.execute("""INSERT INTO prediction_history
            (created_at, input_data, prediction, prediction_label, probabilities, best_split, best_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?)""", (
            datetime.now().astimezone().isoformat(timespec="seconds"),
            json.dumps(input_data, ensure_ascii=False), int(result["prediction"]),
            result["prediction_label"], json.dumps(result.get("probabilities", {})),
            artifact.get("best_split"), artifact.get("best_accuracy")))


def read_history(limit=1000):
    init_db()
    with sqlite3.connect(DB_PATH, timeout=10) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM prediction_history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    records = []
    for row in rows:
        item = dict(row)
        item["input_data"] = json.loads(item["input_data"])
        item["probabilities"] = json.loads(item["probabilities"] or "{}")
        records.append(item)
    return records


def delete_history(record_id):
    with sqlite3.connect(DB_PATH, timeout=10) as con:
        cur = con.execute("DELETE FROM prediction_history WHERE id = ?", (int(record_id),))
        return cur.rowcount > 0


def run_prediction(input_data, artifact):
    model = artifact["model"]
    X, _ = encode_input(input_data, artifact["feature_names"])
    prediction = int(model.predict(X)[0])
    probabilities = {}
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)[0]
        probabilities = {str(int(cls)): float(p) for cls, p in zip(model.classes_, raw)}
    return {
        "prediction": prediction,
        "prediction_label": TARGET_LABELS.get(prediction, "Kelas tidak dikenal"),
        "probabilities": probabilities,
        "best_split": artifact.get("best_split"),
        "best_accuracy": artifact.get("best_accuracy"),
    }, X


def explain_prediction(input_data, artifact):
    model = artifact["model"]
    X, encoded = encode_input(input_data, artifact["feature_names"])
    pred = int(model.predict(X)[0])
    classes = list(model.classes_)
    class_index = classes.index(pred)
    probs = model.predict_proba(X)[0]
    predicted_probability = float(probs[class_index])
    explanation = shap.TreeExplainer(model)(X)
    values = np.asarray(explanation.values, dtype=float)
    n_features, n_classes = len(artifact["feature_names"]), len(classes)
    if values.ndim == 3 and values.shape == (1, n_features, n_classes):
        shap_values = values[0, :, class_index]
    elif values.ndim == 3 and values.shape == (n_classes, 1, n_features):
        shap_values = values[class_index, 0, :]
    elif values.ndim == 2 and values.shape == (1, n_features):
        shap_values = values[0]
    else:
        raise ValueError(f"Format nilai SHAP tidak dikenali: {values.shape}")
    base = np.asarray(explanation.base_values, dtype=float)
    if base.ndim == 0:
        base_value = float(base)
    elif base.ndim == 1 and base.size == n_classes:
        base_value = float(base[class_index])
    elif base.ndim == 1 and base.size == 1:
        base_value = float(base[0])
    elif base.ndim == 2 and base.shape == (1, n_classes):
        base_value = float(base[0, class_index])
    elif base.ndim == 2 and base.shape == (1, 1):
        base_value = float(base[0, 0])
    else:
        raise ValueError(f"Format base value SHAP tidak dikenali: {base.shape}")
    reconstructed = float(base_value + np.sum(shap_values))
    valid = bool(np.isclose(reconstructed, predicted_probability, rtol=1e-4, atol=1e-5))
    rows = []
    for feature, value in zip(artifact["feature_names"], shap_values):
        rows.append({
            "Fitur": FEATURE_LABELS.get(feature, feature),
            "Nama fitur": feature,
            "Nilai input": input_data[feature],
            "Nilai SHAP": float(value),
            "Kontribusi absolut": abs(float(value)),
        })
    df = pd.DataFrame(rows).sort_values("Kontribusi absolut", ascending=False)
    return df, {"base_value": base_value, "shap_sum": float(np.sum(shap_values)),
                "reconstructed_output": reconstructed, "predicted_probability": predicted_probability,
                "probability_error": abs(reconstructed - predicted_probability),
                "probability_additivity_valid": valid}


def show_header():
    st.markdown("<div style='padding:1.2rem 0 .3rem'><span style='color:#2563eb;font-weight:800;letter-spacing:.12em'>MACHINE LEARNING · RANDOM FOREST · SHAP</span><h1 style='margin:.25rem 0;color:#142b4a'>💙 SaltCare</h1><p style='color:#64748b'>Kenali Risiko Hipertensi, Mulai dari Pola Konsumsi Garam.</p></div>", unsafe_allow_html=True)
    st.divider()


def page_home(artifact):
    st.title("Kenali Risiko Hipertensi")
    st.markdown("SaltCare merupakan prototipe penelitian untuk mempelajari klasifikasi hipertensi berdasarkan faktor gaya hidup dan kesehatan. Hasil model dilengkapi penjelasan SHAP untuk membantu memahami kontribusi fitur terhadap prediksi.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Algoritma", "Random Forest")
    c2.metric("Jumlah fitur", len(artifact["feature_names"]))
    c3.metric("Akurasi tersimpan", f"{artifact['best_accuracy']:.2%}" if artifact.get("best_accuracy") is not None else "Tidak tersedia")
    st.info("Hasil merupakan keluaran model penelitian, bukan diagnosis medis. Jangan gunakan hasil ini sebagai pengganti pemeriksaan atau saran tenaga kesehatan.")
    st.markdown("### Mulai dari sini")
    st.write("Pilih **Prediksi** di menu navigasi untuk memasukkan data, melihat hasil klasifikasi, dan membuka analisis SHAP.")


def page_prediction(artifact):
    st.title("Prediksi Risiko Hipertensi")
    st.write("Lengkapi sepuluh fitur berikut. Pastikan nilai dan kategori sesuai dengan data yang digunakan saat pelatihan model.")
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=35, step=1)
            salt = st.number_input("Konsumsi garam (sesuai satuan dataset)", min_value=0.0, max_value=100.0, value=7.0, step=0.1)
            stress = st.number_input("Skor stres (1–10)", min_value=1, max_value=10, value=5, step=1)
            bp = st.selectbox("Riwayat tekanan darah", ["Normal", "Prehypertension", "Hypertension"])
            sleep = st.number_input("Durasi tidur (jam per hari)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        with right:
            bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=1.0, max_value=100.0, value=24.0, step=0.1)
            medication = st.selectbox("Penggunaan obat", ["None", "ACE Inhibitor", "Beta Blocker", "Diuretic", "Other"])
            family = st.selectbox("Riwayat hipertensi keluarga", ["No", "Yes"])
            exercise = st.selectbox("Tingkat aktivitas olahraga", ["Low", "Moderate", "High"])
            smoking = st.selectbox("Status merokok", ["Non-Smoker", "Smoker"])
        submitted = st.form_submit_button("🔎 Prediksi Sekarang", type="primary", use_container_width=True)
    if submitted:
        input_data = {"Age": int(age), "Salt_Intake": float(salt), "Stress_Score": int(stress),
                      "BP_History": bp, "Sleep_Duration": float(sleep), "BMI": float(bmi),
                      "Medication": medication, "Family_History": family,
                      "Exercise_Level": exercise, "Smoking_Status": smoking}
        try:
            result, _ = run_prediction(input_data, artifact)
            save_history(input_data, result, artifact)
            st.session_state["last_input"] = input_data
            st.session_state["last_result"] = result
            st.session_state.pop("last_shap", None)
            st.success("Prediksi berhasil dijalankan dan dicatat pada riwayat.")
        except Exception as exc:
            st.error(f"Prediksi gagal: {exc}")
    if st.session_state.get("last_result"):
        result = st.session_state["last_result"]
        st.divider()
        st.subheader("Hasil Prediksi")
        if result["prediction"] == 1:
            st.warning(f"Hasil model: **{result['prediction_label']}**")
        else:
            st.success(f"Hasil model: **{result['prediction_label']}**")
        probs = result.get("probabilities", {})
        if probs:
            pcols = st.columns(len(probs))
            for col, (cls, prob) in zip(pcols, probs.items()):
                col.metric(TARGET_LABELS.get(int(cls), f"Kelas {cls}"), f"{prob:.1%}")
        if result.get("best_split"):
            st.caption(f"Skenario split model tersimpan: {result['best_split']} · Akurasi evaluasi tersimpan: {result.get('best_accuracy', 0):.2%}")
        st.info("Ini adalah klasifikasi model penelitian, bukan diagnosis medis.")
        if st.button("Tampilkan penjelasan SHAP", type="primary"):
            with st.spinner("Menghitung penjelasan SHAP..."):
                try:
                    shap_df, validation = explain_prediction(st.session_state["last_input"], artifact)
                    st.session_state["last_shap"] = (shap_df, validation)
                except Exception as exc:
                    st.error(f"Analisis SHAP gagal: {exc}")
        if st.session_state.get("last_shap"):
            shap_df, validation = st.session_state["last_shap"]
            st.subheader("Interpretasi Hasil dengan SHAP")
            st.caption("Fitur diurutkan berdasarkan besarnya kontribusi absolut. Tanda positif/negatif harus ditafsirkan pada skala keluaran yang dijelaskan oleh model.")
            chart_df = shap_df.set_index("Fitur")["Nilai SHAP"].sort_values()
            st.bar_chart(chart_df, horizontal=True)
            st.dataframe(shap_df[["Fitur", "Nilai input", "Nilai SHAP"]], use_container_width=True, hide_index=True)
            if validation["probability_additivity_valid"]:
                st.success("Pemeriksaan additivity: base value + kontribusi SHAP cocok dengan probabilitas kelas prediksi.")
            else:
                st.warning("Pemeriksaan additivity: penjumlahan SHAP belum cocok dengan probabilitas kelas prediksi. Skala output tidak boleh dianggap terverifikasi.")
            with st.expander("Detail validasi teknis SHAP"):
                st.json(validation)


def page_dashboard():
    st.title("Dashboard Analytics")
    records = read_history(limit=100000)
    total = len(records)
    positive = sum(int(r["prediction"]) == 1 for r in records)
    negative = sum(int(r["prediction"]) == 0 for r in records)
    a, b, c = st.columns(3)
    a.metric("Total prediksi tersimpan", total)
    b.metric("Terklasifikasi hipertensi", positive)
    c.metric("Tidak terklasifikasi hipertensi", negative)
    if total:
        st.subheader("Distribusi hasil prediksi")
        st.bar_chart(pd.DataFrame({"Jumlah": [negative, positive]}, index=["Kelas 0", "Kelas 1"]))
        st.subheader("Lima prediksi terbaru")
        st.dataframe(history_frame(records[:5]), use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada riwayat prediksi. Jalankan prediksi terlebih dahulu.")


def history_frame(records):
    rows = []
    for r in records:
        d = r["input_data"]
        probs = r.get("probabilities", {})
        rows.append({"ID": r["id"], "Waktu": r["created_at"].replace("T", " "),
                     "Usia": d.get("Age"), "Garam": d.get("Salt_Intake"), "BMI": d.get("BMI"),
                     "Prediksi": r["prediction_label"],
                     "Probabilitas kelas 1": f"{probs.get('1', 0):.1%}" if probs else "—"})
    return pd.DataFrame(rows)


def page_history():
    st.title("Riwayat Prediksi")
    records = read_history(limit=1000)
    if not records:
        st.info("Belum ada riwayat prediksi.")
        return
    st.dataframe(history_frame(records), use_container_width=True, hide_index=True)
    with st.expander("Lihat detail input pada suatu riwayat"):
        ids = [r["id"] for r in records]
        selected = st.selectbox("Pilih ID riwayat", ids)
        item = next(r for r in records if r["id"] == selected)
        st.json({"input_data": item["input_data"], "probabilities": item["probabilities"], "hasil": item["prediction_label"]})
    with st.expander("Hapus riwayat"):
        delete_id = st.selectbox("ID yang akan dihapus", [r["id"] for r in records], key="delete_id")
        confirm = st.checkbox("Saya yakin ingin menghapus catatan ini.")
        if st.button("Hapus catatan", disabled=not confirm):
            if delete_history(delete_id):
                st.success(f"Riwayat ID {delete_id} dihapus.")
                st.rerun()
            else:
                st.error("Catatan tidak ditemukan.")


def page_education():
    st.title("Edukasi: Garam dan Hipertensi")
    st.markdown("""
SaltCare adalah prototipe pembelajaran yang mengklasifikasikan data berdasarkan sepuluh fitur masukan. Hasilnya mencerminkan pola yang dipelajari model dari data pelatihan, bukan penilaian klinis individual.

- **Konsumsi garam:** perhatikan jumlah garam dari makanan dan bumbu yang dikonsumsi.
- **Aktivitas fisik dan tidur:** keduanya merupakan bagian dari pola hidup yang dapat berkaitan dengan kesehatan.
- **Riwayat tekanan darah dan keluarga:** masukan ini mengikuti kategori yang digunakan pada dataset model.
- **Penggunaan obat:** masukkan kategori yang sesuai; jangan mengubah atau menghentikan obat berdasarkan hasil aplikasi.

Untuk mengetahui tekanan darah, lakukan pengukuran dengan alat yang sesuai dan konsultasikan hasilnya kepada tenaga kesehatan.
""")
    st.warning("Aplikasi ini bukan alat diagnosis medis dan tidak menggantikan konsultasi dengan tenaga kesehatan.")


def page_about(artifact):
    st.title("Tentang SaltCare")
    st.markdown("SaltCare merupakan prototipe penelitian untuk menunjukkan alur klasifikasi risiko hipertensi menggunakan Random Forest dan interpretasi lokal dengan SHAP.")
    st.subheader("Informasi model yang dimuat")
    st.json({"jenis_model": type(artifact["model"]).__name__, "target": artifact.get("target_name"),
             "fitur": artifact.get("feature_names"), "jumlah_fitur": len(artifact.get("feature_names", [])),
             "skenario_split": artifact.get("best_split"), "akurasi_tersimpan": artifact.get("best_accuracy")})
    st.caption("Metrik di atas dibaca dari artefak model yang tersedia; bukan hasil evaluasi ulang oleh aplikasi ini.")
    st.info("SHAP menjelaskan kontribusi fitur terhadap keluaran model, bukan membuktikan hubungan sebab-akibat medis.")


def main():
    show_header()
    try:
        artifact = load_artifact()
    except Exception as exc:
        st.error(f"Model gagal dimuat: {exc}")
        st.stop()
    with st.sidebar:
        st.markdown("# 💙 SaltCare")
        page = st.radio("Navigasi", ["Beranda", "Prediksi", "Dashboard", "Riwayat", "Edukasi", "Tentang"])
        st.divider()
        st.caption("Prototipe penelitian · Random Forest + SHAP")
    if page == "Beranda": page_home(artifact)
    elif page == "Prediksi": page_prediction(artifact)
    elif page == "Dashboard": page_dashboard()
    elif page == "Riwayat": page_history()
    elif page == "Edukasi": page_education()
    elif page == "Tentang": page_about(artifact)

if __name__ == "__main__":
    main()
