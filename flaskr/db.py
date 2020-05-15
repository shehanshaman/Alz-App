import sqlite3

import click
import os
from flask import current_app
from flask import g
from flask.cli import with_appcontext
from pathlib import Path
import pandas as pd

ROOT_PATH = Path.cwd()
MAIL_PATH = ROOT_PATH / "flaskr" / "mail"
ANNOTATION_TBL = ROOT_PATH / "flaskr" / "upload" / "AnnotationTbls"
UPLOAD_FOLDER = ROOT_PATH / "flaskr" / "upload"

def get_db():
    """Connect to the application's configured database. The connection
    is unique for each request and will be reused if this is called
    again.
    """
    if "db" not in g:
        g.db = sqlite3.connect(
            current_app.config["DATABASE"], detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db


def close_db(e=None):
    """If this request connected to the database, close the
    connection.
    """
    db = g.pop("db", None)

    if db is not None:
        db.close()


def init_db():
    """Clear existing data and create new tables."""
    db = get_db()

    with current_app.open_resource("schema.sql") as f:
        db.executescript(f.read().decode("utf8"))


@click.command("init-db")
@with_appcontext
def init_db_command():
    """Clear existing data and create new tables."""
    init_db()
    add_mail_templates()
    add_annotation()
    setup_folders()
    sample_pkl_create()
    
    click.echo("Initialized the database.")

def setup_folders():
    #users path added
    user_path = UPLOAD_FOLDER / "users"
    try:
        os.makedirs(user_path)
    except OSError:
        pass

    #Other annotation folder create
    other_path = UPLOAD_FOLDER / "AnnotationTbls" / "other"
    try:
        os.makedirs(other_path)
    except OSError:
        pass

def sample_pkl_create():
    #sample pkl create
    sample_path = UPLOAD_FOLDER / "sample"

    df = pd.read_csv((sample_path / 'GSE5281-GPL570.zip'))
    df = df.set_index(["ID"])
    df.index.name = None
    df.columns.name = "ID"
    df.to_pickle((sample_path /'GSE5281-GPL570.pkl'))

def add_annotation():
    file_names = [f for f in os.listdir(ANNOTATION_TBL) if os.path.isfile((ANNOTATION_TBL / f))]
    db = get_db()
    for file in file_names:
        path = "/AnnotationTbls/" + file
        db.execute(
            "INSERT INTO file (file_name, file_type, path, user_id, is_annotation, has_class) VALUES (?, ?, ?, ?, ?, ?)",
            (file, file.split('.')[1], path, 0, 1, 0),
        )
    db.commit()

def add_mail_templates():
    db = get_db()
    subjects = ["verify", "reset", "infrequent"]
    for subject in subjects:
        file_name = MAIL_PATH / (subject + '_mail.html')
        f = open(file_name, "r")
        message = f.read()
        db.execute(
            "INSERT INTO mail_template (subject, message) VALUES (?, ?)",
            (subject, message),
        )
    db.commit()

def init_app(app):
    """Register database functions with the Flask app. This is called by
    the application factory.
    """
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
