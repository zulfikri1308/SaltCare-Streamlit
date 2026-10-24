
import json
import sqlite3
from datetime import datetime
from flask import current_app


def get_connection():
    connection = sqlite3.connect(
        current_app.config["DATABASE_PATH"],
        timeout=10
    )
    connection.row_factory = sqlite3.Row
    return connection


def init_history_db():
    connection = get_connection()

    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                input_data TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                prediction_label TEXT NOT NULL,
                probabilities TEXT,
                best_split TEXT,
                best_accuracy REAL
            )
        """)
        connection.commit()
    finally:
        connection.close()


def save_prediction(input_data, result):
    connection = get_connection()

    try:
        cursor = connection.execute("""
            INSERT INTO prediction_history (
                created_at,
                input_data,
                prediction,
                prediction_label,
                probabilities,
                best_split,
                best_accuracy
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().astimezone().isoformat(timespec="seconds"),
            json.dumps(input_data, ensure_ascii=False),
            int(result["prediction"]),
            result["prediction_label"],
            json.dumps(result.get("probabilities"), ensure_ascii=False),
            result.get("best_split"),
            result.get("best_accuracy"),
        ))

        connection.commit()
        return cursor.lastrowid
    finally:
        connection.close()


def get_prediction_history(limit=100):
    connection = get_connection()

    try:
        rows = connection.execute("""
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

        history = []

        for row in rows:
            item = dict(row)
            item["input_data"] = json.loads(item["input_data"])
            item["probabilities"] = (
                json.loads(item["probabilities"])
                if item["probabilities"]
                else {}
            )
            history.append(item)

        return history
    finally:
        connection.close()


def delete_prediction(history_id):
    connection = get_connection()

    try:
        cursor = connection.execute(
            "DELETE FROM prediction_history WHERE id = ?",
            (history_id,)
        )
        connection.commit()
        return cursor.rowcount > 0
    finally:
        connection.close()
