"""Helpers for working with a SQLite database via a global path."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence
import os





def _resolve_path(db_name: str | Path) -> Path:
	return Path(db_name).expanduser().resolve()


def select(query: str, parameters: Sequence[Any] | None = None) -> list[dict[str, Any]]:
	"""Run a SELECT query and return the results as a list of dictionaries."""

	db_path = _resolve_path(os.getenv("DB_NAME", "undefined"))
	params: Iterable[Any] = parameters or []
	with sqlite3.connect(db_path) as connection:
		connection.row_factory = sqlite3.Row
		rows = connection.execute(query, params).fetchall()
	return [dict(row) for row in rows]


def insert_or_update(db_name: str | Path, query: str, parameters: Sequence[Any] | None = None) -> int:
	"""Execute an INSERT or UPDATE statement and return the affected row count."""

	db_path = _resolve_path(os.getenv("DB_NAME", "undefined"))
	params: Iterable[Any] = parameters or []
	with sqlite3.connect(db_path) as connection:
		cursor = connection.execute(query, params)
		connection.commit()
		return cursor.rowcount


def get_db_schema(db_name: str | Path) -> dict[str, list[dict[str, Any]]]:
	"""Return the database schema keyed by table name."""

	db_path = _resolve_path(db_name)
	with sqlite3.connect(db_path) as connection:
		connection.row_factory = sqlite3.Row
		tables = connection.execute(
			"SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
		).fetchall()
		schema: dict[str, list[dict[str, Any]]] = {}
		for table in tables:
			columns = connection.execute(f"PRAGMA table_info({table['name']})").fetchall()
			schema[table["name"]] = [
				{
					"cid": column["cid"],
					"name": column["name"],
					"type": column["type"],
					"notnull": bool(column["notnull"]),
					"default": column["dflt_value"],
					"primary_key": bool(column["pk"]),
				}
				for column in columns
			]
	return schema

