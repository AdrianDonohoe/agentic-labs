# Contacts demo (Flask + SQLite)

Simple demo web application showing a contact list and a form to add contacts. The app uses Flask with server-side rendering (Jinja templates) and stores data in a local SQLite database.

Features
- List contacts (HTML table)
- Add contact (form)
- Bootstrap styling and a top navigation bar

Quick start
1. Create and activate a Python virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python3 web_app.py
```

4. Open your browser at http://127.0.0.1:5000

Notes
- The SQLite database file `contacts.db` will be created automatically in the project root when the app first starts.
