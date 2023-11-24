# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import sqlite3

# Create a database connection
con = sqlite3.connect("nmn.db")
con.execute("PRAGMA foreign_keys = ON")
con.executescript(
    """
CREATE TABLE IF NOT EXISTS ComponentCategories (
    id INTEGER PRIMARY KEY,
    name TEXT,
    content BLOB
);
CREATE TABLE IF NOT EXISTS Components (
    id INTEGER PRIMARY KEY,
    category TEXT,
    score_id INTEGER,
    content BLOB,
    FOREIGN KEY (category) REFERENCES ComponentCategories(name),
    FOREIGN KEY (score_id) REFERENCES Scores(id)
);
CREATE TABLE IF NOT EXISTS Scores (
    id INTEGER PRIMARY KEY,
    content BLOB
);
"""
)

# %%
from pathlib import Path
import sys

lib_path = str(Path("../lib").resolve())
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# %%
from component_classification import (
    generate_categories,
    generate_components,
    classify_components,
)

paths, categories = generate_categories()
components = generate_components()
results = classify_components(components, categories)

# %%
# add categories to database

categories_table = [
    (path.name, category.tobytes()) for path, category in zip(paths, categories)
]

with con:
    con.executemany(
        "INSERT INTO ComponentCategories (name, content) VALUES (?, ?)",
        categories_table,
    )

# %%
components_table = [(paths[res].name,) for component, res in zip(components, results)]

# %%

# %%
