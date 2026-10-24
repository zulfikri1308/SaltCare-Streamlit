
import os
import joblib
import pandas as pd
from flask import current_app


# Pemetaan harus konsisten dengan notebook pelatihan.
CATEGORY_MAPPINGS = {
    "BP_History": {
        "Normal": 0,
        "Prehypertension": 1,
        "Hypertension": 2,
    },
    "Exercise_Level": {
        "Low": 0,
        "Moderate": 1,
        "High": 2,
    },
    "Family_History": {
        "No": 0,
        "Yes": 1,
    },
    "Smoking_Status": {
        "Non-Smoker": 0,
        "Smoker": 1,
    },
    "Medication": {
        "ACE Inhibitor": 0,
        "Beta Blocker": 1,
        "Diuretic": 2,
        "None": 3,
        "Other": 4,
    },
}

TARGET_LABELS = {
    0: "Tidak terklasifikasi hipertensi",
    1: "Terklasifikasi hipertensi",
}


def load_model_artifact():
    model_path = current_app.config["MODEL_PATH"]

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"File model tidak ditemukan: {model_path}"
        )

    artifact = joblib.load(model_path)

    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Format artefak model tidak valid.")

    return artifact


def get_model_info():
    artifact = load_model_artifact()
    model = artifact["model"]

    return {
        "model_type": type(model).__name__,
        "best_split": artifact.get("best_split"),
        "best_accuracy": artifact.get("best_accuracy"),
        "feature_names": artifact.get("feature_names", []),
        "n_features_in": getattr(model, "n_features_in_", None),
        "model_classes": getattr(model, "classes_", None),
        "target_name": artifact.get("target_name"),
    }


def predict_hypertension(input_data):
    artifact = load_model_artifact()
    model = artifact["model"]
    feature_names = artifact["feature_names"]

    encoded_data = input_data.copy()

    # Ubah kategori menjadi angka sesuai preprocessing notebook.
    for feature, mapping in CATEGORY_MAPPINGS.items():
        if feature in encoded_data:
            category = encoded_data[feature]

            if category not in mapping:
                raise ValueError(
                    f"Kategori {category!r} tidak valid untuk {feature}."
                )

            encoded_data[feature] = mapping[category]

    # Pastikan urutan fitur sama dengan urutan saat pelatihan.
    missing_features = [
        name for name in feature_names
        if name not in encoded_data
    ]

    if missing_features:
        raise ValueError(
            f"Fitur belum tersedia: {missing_features}"
        )

    input_df = pd.DataFrame(
        [[encoded_data[name] for name in feature_names]],
        columns=feature_names,
    )

    prediction = int(model.predict(input_df)[0])

    probabilities = None
    if hasattr(model, "predict_proba"):
        raw_probabilities = model.predict_proba(input_df)[0]
        probabilities = {
            str(int(class_id)): float(probability)
            for class_id, probability in zip(
                model.classes_, raw_probabilities
            )
        }

    return {
        "prediction": prediction,
        "prediction_label": TARGET_LABELS.get(
            prediction, "Kelas tidak dikenal"
        ),
        "probabilities": probabilities,
        "best_split": artifact.get("best_split"),
        "best_accuracy": artifact.get("best_accuracy"),
    }
