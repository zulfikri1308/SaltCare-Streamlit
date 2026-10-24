
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
)
from services.model_service import predict_hypertension
from services.shap_service import explain_prediction
from services.history_service import (
    save_prediction,
    get_prediction_history,
    delete_prediction,
)

prediction_bp = Blueprint("prediction", __name__)

FEATURES = [
    {
        "name": "Age",
        "label": "Usia",
        "type": "number",
        "min": 1,
        "max": 120,
        "step": 1,
        "unit": "tahun",
    },
    {
        "name": "Salt_Intake",
        "label": "Konsumsi Garam",
        "type": "number",
        "min": 0,
        "max": 100,
        "step": "any",
        "unit": "sesuai satuan dataset",
    },
    {
        "name": "Stress_Score",
        "label": "Skor Stres",
        "type": "number",
        "min": 1,
        "max": 10,
        "step": 1,
        "unit": "skala 1–10",
    },
    {
        "name": "BP_History",
        "label": "Riwayat Tekanan Darah",
        "type": "select",
        "options": ["Normal", "Prehypertension", "Hypertension"],
    },
    {
        "name": "Sleep_Duration",
        "label": "Durasi Tidur",
        "type": "number",
        "min": 0,
        "max": 24,
        "step": "any",
        "unit": "jam per hari",
    },
    {
        "name": "BMI",
        "label": "Indeks Massa Tubuh (BMI)",
        "type": "number",
        "min": 1,
        "max": 100,
        "step": "any",
        "unit": "kg/m²",
    },
    {
        "name": "Medication",
        "label": "Penggunaan Obat",
        "type": "select",
        "options": [
            "None",
            "ACE Inhibitor",
            "Beta Blocker",
            "Diuretic",
            "Other",
        ],
    },
    {
        "name": "Family_History",
        "label": "Riwayat Hipertensi Keluarga",
        "type": "select",
        "options": ["No", "Yes"],
    },
    {
        "name": "Exercise_Level",
        "label": "Tingkat Aktivitas Olahraga",
        "type": "select",
        "options": ["Low", "Moderate", "High"],
    },
    {
        "name": "Smoking_Status",
        "label": "Status Merokok",
        "type": "select",
        "options": ["Non-Smoker", "Smoker"],
    },
]


@prediction_bp.route("/prediksi", methods=["GET", "POST"])
def prediction():
    if request.method == "GET":
        return render_template("prediction.html", features=FEATURES)

    input_data = {}

    try:
        for feature in FEATURES:
            name = feature["name"]
            label = feature["label"]
            value = request.form.get(name, "").strip()

            if not value:
                raise ValueError(f"Kolom {label} wajib diisi.")

            if feature["type"] == "number":
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Nilai {label} harus berupa angka."
                    )

                if number != number or abs(number) == float("inf"):
                    raise ValueError(f"Nilai {label} tidak valid.")

                if not feature["min"] <= number <= feature["max"]:
                    raise ValueError(
                        f"{label} harus berada antara "
                        f"{feature['min']} dan {feature['max']}."
                    )

                if feature["step"] == 1 and not number.is_integer():
                    raise ValueError(
                        f"{label} harus berupa bilangan bulat."
                    )

                input_data[name] = (
                    int(number) if number.is_integer() else number
                )

            else:
                if value not in feature["options"]:
                    raise ValueError(
                        f"Pilihan untuk {label} tidak valid."
                    )

                input_data[name] = value

        result_data = predict_hypertension(input_data)

        session["saltcare_input"] = input_data
        session["saltcare_result"] = result_data

        save_prediction(input_data, result_data)

        return render_template(
            "result.html",
            result=result_data,
            input_data=input_data,
        )

    except (ValueError, KeyError, FileNotFoundError) as error:
        flash(str(error), "danger")
        return redirect(url_for("prediction.prediction"))


@prediction_bp.route("/hasil", methods=["GET"])
def result():
    result_data = session.get("saltcare_result")
    input_data = session.get("saltcare_input")

    if not result_data or not input_data:
        flash(
            "Silakan lakukan prediksi terlebih dahulu.",
            "info",
        )
        return redirect(url_for("prediction.prediction"))

    return render_template(
        "result.html",
        result=result_data,
        input_data=input_data,
    )


@prediction_bp.route("/analisis", methods=["GET"])
def explanation():
    input_data = session.get("saltcare_input")

    if not input_data:
        flash(
            "Silakan lakukan prediksi terlebih dahulu "
            "sebelum membuka analisis SHAP.",
            "info",
        )
        return redirect(url_for("prediction.prediction"))

    try:
        analysis = explain_prediction(input_data)

        return render_template(
            "explanation.html",
            analysis=analysis,
            input_data=input_data,
        )

    except (ValueError, KeyError, FileNotFoundError) as error:
        flash(f"Analisis SHAP gagal: {error}", "danger")
        return redirect(url_for("prediction.result"))


@prediction_bp.route("/riwayat", methods=["GET"])
def history():
    records = get_prediction_history(limit=100)

    total_records = len(records)
    hypertension_count = sum(
        1 for item in records if item["prediction"] == 1
    )
    non_hypertension_count = sum(
        1 for item in records if item["prediction"] == 0
    )

    return render_template(
        "history.html",
        records=records,
        total_records=total_records,
        hypertension_count=hypertension_count,
        non_hypertension_count=non_hypertension_count,
    )


@prediction_bp.route(
    "/riwayat/hapus/<int:history_id>",
    methods=["POST"],
)
def delete_history(history_id):
    try:
        deleted = delete_prediction(history_id)

        if deleted:
            flash(
                "Riwayat prediksi berhasil dihapus.",
                "success",
            )
        else:
            flash(
                "Riwayat tidak ditemukan atau sudah dihapus.",
                "warning",
            )

    except Exception:
        flash(
            "Riwayat gagal dihapus karena terjadi kesalahan database.",
            "danger",
        )

    return redirect(url_for("prediction.history"))
