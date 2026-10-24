
import numpy as np
import pandas as pd
import shap

from services.model_service import (
    load_model_artifact,
    CATEGORY_MAPPINGS,
    TARGET_LABELS,
)


FEATURE_LABELS = {
    "Age": "Usia",
    "Salt_Intake": "Konsumsi garam",
    "Stress_Score": "Skor stres",
    "BP_History": "Riwayat tekanan darah",
    "Sleep_Duration": "Durasi tidur",
    "BMI": "Indeks massa tubuh (BMI)",
    "Medication": "Penggunaan obat",
    "Family_History": "Riwayat hipertensi keluarga",
    "Exercise_Level": "Tingkat aktivitas olahraga",
    "Smoking_Status": "Status merokok",
}


def _extract_class_values(raw_values, class_index, n_features, n_classes):
    """Ambil nilai SHAP untuk satu sampel dan satu kelas."""
    values = np.asarray(raw_values, dtype=float)

    if values.ndim == 3:
        # Format SHAP modern: (sampel, fitur, kelas).
        if values.shape[0] == 1 and values.shape[1] == n_features:
            if values.shape[2] != n_classes:
                raise ValueError(
                    "Dimensi kelas pada keluaran SHAP tidak cocok dengan model."
                )
            values = values[0, :, class_index]

        # Format alternatif: (kelas, sampel, fitur).
        elif (
            values.shape[0] == n_classes
            and values.shape[1] == 1
            and values.shape[2] == n_features
        ):
            values = values[class_index, 0, :]

        else:
            raise ValueError(
                f"Format nilai SHAP tidak dikenali: {values.shape}"
            )

    elif values.ndim == 2:
        # Format satu kelas: (sampel, fitur).
        if values.shape == (1, n_features):
            values = values[0]
        else:
            raise ValueError(
                f"Dimensi nilai SHAP tidak sesuai: {values.shape}"
            )

    elif values.ndim == 1:
        if values.size != n_features:
            raise ValueError("Jumlah nilai SHAP tidak sesuai dengan fitur.")
    else:
        raise ValueError(
            f"Jumlah dimensi nilai SHAP tidak didukung: {values.ndim}"
        )

    values = np.asarray(values, dtype=float).reshape(-1)

    if values.size != n_features:
        raise ValueError(
            "Jumlah kontribusi SHAP tidak sama dengan jumlah fitur model."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("Nilai SHAP mengandung angka yang tidak valid.")

    return values


def _extract_base_value(base_values, class_index, n_classes):
    """Ambil base value untuk kelas yang dijelaskan."""
    base = np.asarray(base_values, dtype=float)

    if base.ndim == 0:
        return float(base)

    if base.ndim == 1:
        if base.size == n_classes:
            return float(base[class_index])
        if base.size == 1:
            return float(base[0])

    if base.ndim == 2:
        if base.shape[0] == 1 and base.shape[1] == n_classes:
            return float(base[0, class_index])
        if base.shape == (1, 1):
            return float(base[0, 0])

    raise ValueError(
        f"Format base value SHAP tidak dikenali: {base.shape}"
    )


def explain_prediction(input_data):
    """
    Menjelaskan satu prediksi Random Forest dengan SHAP.

    Memeriksa apakah base value + seluruh kontribusi SHAP konsisten
    dengan probabilitas kelas yang dikeluarkan model. Jika tidak cocok,
    hasil dilaporkan sebagai belum tervalidasi pada skala output tersebut.
    """
    artifact = load_model_artifact()
    model = artifact["model"]
    feature_names = artifact["feature_names"]

    original_data = input_data.copy()
    encoded_data = input_data.copy()

    # Encoding harus konsisten dengan model_service.py.
    for feature, mapping in CATEGORY_MAPPINGS.items():
        if feature in encoded_data:
            category = encoded_data[feature]

            if category not in mapping:
                raise ValueError(
                    f"Kategori {category!r} tidak valid untuk {feature}."
                )

            encoded_data[feature] = mapping[category]

    missing_features = [
        name for name in feature_names if name not in encoded_data
    ]
    if missing_features:
        raise ValueError(
            f"Fitur belum tersedia: {missing_features}"
        )

    X = pd.DataFrame(
        [[encoded_data[name] for name in feature_names]],
        columns=feature_names,
    )

    prediction = int(model.predict(X)[0])
    model_classes = list(model.classes_)
    n_classes = len(model_classes)

    if prediction not in model_classes:
        raise ValueError("Kelas prediksi tidak ditemukan pada model.")

    class_index = model_classes.index(prediction)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model tidak menyediakan predict_proba().")

    probabilities = model.predict_proba(X)[0]
    predicted_probability = float(probabilities[class_index])

    # Menggunakan Explanation agar nilai dan base value didapat bersama.
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    shap_values = _extract_class_values(
        raw_values=explanation.values,
        class_index=class_index,
        n_features=len(feature_names),
        n_classes=n_classes,
    )

    base_value = _extract_base_value(
        explanation.base_values,
        class_index=class_index,
        n_classes=n_classes,
    )

    shap_sum = float(np.sum(shap_values))
    reconstructed_output = float(base_value + shap_sum)

    # Periksa apakah hasil rekonstruksi cocok dengan probabilitas kelas.
    probability_error = abs(
        reconstructed_output - predicted_probability
    )
    probability_additivity_valid = bool(
        np.isclose(
            reconstructed_output,
            predicted_probability,
            rtol=1e-4,
            atol=1e-5,
        )
    )

    # Random Forest SHAP biasanya menjelaskan probabilitas pada
    # konfigurasi default ini, tetapi skala harus diverifikasi.
    output_scale = (
        "probability"
        if probability_additivity_valid
        else "unverified"
    )

    contributions = []
    for feature, shap_value in zip(feature_names, shap_values):
        contributions.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "value": original_data[feature],
            "model_value": encoded_data[feature],
            "shap_value": float(shap_value),
        })

    contributions.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )

    return {
        "prediction": prediction,
        "prediction_label": TARGET_LABELS.get(
            prediction, "Kelas tidak dikenal"
        ),
        "predicted_probability": predicted_probability,
        "contributions": contributions,
        "salt_intake": original_data.get("Salt_Intake"),
        "feature_count": len(contributions),
        "explanation_type": "local",

        # Data validasi untuk pemeriksaan teknis.
        "validation": {
            "base_value": base_value,
            "shap_sum": shap_sum,
            "reconstructed_output": reconstructed_output,
            "predicted_probability": predicted_probability,
            "probability_error": probability_error,
            "probability_additivity_valid": probability_additivity_valid,
            "output_scale": output_scale,
        },
    }
