import os
import sqlite3
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for


# Paths
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "contacts.db"


def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    # create the database and table if it doesn't exist
    conn = get_db_connection()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT NOT NULL
            )
            """
        )
    conn.close()


app = Flask(__name__)
app.config["DATABASE"] = str(DB_PATH)


@app.route("/")
def index():
    return redirect(url_for("list_contacts"))


@app.route("/contacts")
def list_contacts():
    conn = get_db_connection()
    cur = conn.execute("SELECT id, name, email, phone FROM contacts ORDER BY id DESC")
    contacts = cur.fetchall()
    conn.close()
    return render_template("list_contacts.html", contacts=contacts)


@app.route("/add", methods=["GET", "POST"])
def add_contact():
    error = None
    values = {"name": "", "email": "", "phone": ""}
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        values = {"name": name, "email": email, "phone": phone}
        if not name or not email or not phone:
            error = "All fields are required."
        else:
            conn = get_db_connection()
            with conn:
                conn.execute(
                    "INSERT INTO contacts (name, email, phone) VALUES (?, ?, ?)",
                    (name, email, phone),
                )
            conn.close()
            return redirect(url_for("list_contacts"))

    return render_template("add_contact.html", error=error, values=values)


if __name__ == "__main__":
    # ensure the DB exists
    init_db()
    # run the app
    app.run(host="127.0.0.1", port=5000, debug=True)
