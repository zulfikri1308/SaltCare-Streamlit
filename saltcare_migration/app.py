
from flask import Flask
from config import Config
from routes.main import main_bp
from routes.prediction import prediction_bp
from services.history_service import init_history_db


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Pastikan folder database tersedia.
    import os
    os.makedirs(
        os.path.dirname(app.config["DATABASE_PATH"]),
        exist_ok=True
    )

    # Inisialisasi tabel riwayat.
    with app.app_context():
        init_history_db()

    app.register_blueprint(main_bp)
    app.register_blueprint(prediction_bp)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
