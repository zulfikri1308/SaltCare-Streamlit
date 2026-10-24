
from flask import Blueprint, render_template
from services.history_service import get_prediction_history

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/dashboard")
def dashboard():
    # Ambil riwayat prediksi dari database SQLite.
    records = get_prediction_history(limit=100000)

    # Hitung statistik berdasarkan hasil prediksi tersimpan.
    total_predictions = len(records)

    hypertension_count = sum(
        1 for record in records
        if int(record["prediction"]) == 1
    )

    non_hypertension_count = sum(
        1 for record in records
        if int(record["prediction"]) == 0
    )

    # Hitung persentase distribusi hasil.
    if total_predictions > 0:
        hypertension_percentage = round(
            hypertension_count / total_predictions * 100, 1
        )
        non_hypertension_percentage = round(
            non_hypertension_count / total_predictions * 100, 1
        )
    else:
        hypertension_percentage = 0
        non_hypertension_percentage = 0

    # Lima prediksi terbaru untuk ditampilkan di dashboard.
    latest_records = records[:5]

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        hypertension_count=hypertension_count,
        non_hypertension_count=non_hypertension_count,
        hypertension_percentage=hypertension_percentage,
        non_hypertension_percentage=non_hypertension_percentage,
        latest_records=latest_records,
    )


@main_bp.route("/edukasi")
def education():
    return render_template("education.html")


@main_bp.route("/tentang")
def about():
    return render_template("about.html")
