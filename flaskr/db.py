import sqlite3

import click
import os
from flask import current_app
from flask import g
from flask.cli import with_appcontext
from pathlib import Path
import pandas as pd
from flaskr import auth

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pickle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

ROOT_PATH = Path.cwd()
MAIL_PATH = ROOT_PATH / "flaskr" / "mail"
ANNOTATION_TBL = ROOT_PATH / "flaskr" / "upload" / "AnnotationTbls"
UPLOAD_FOLDER = ROOT_PATH / "flaskr" / "upload"
SAMPLE_PATH = ROOT_PATH / "flaskr" / "upload" / "sample"

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
    create_user()
    create_default_model()

    click.echo("Initialized the database.")

@click.command("init-model")
@with_appcontext
def init_default_model_command():
    create_default_model()

    click.echo("Initialized the default model.")

def create_default_model():
    # path_csv = SAMPLE_PATH / "GSE5281_DE_200.csv"
    path_csv = SAMPLE_PATH / "GeNet_GSE5281-GPL570.csv"
    df = pd.read_csv(path_csv, index_col=0)
    df.columns.name = df.index.name
    df.index.name = None

    # col = ["AC004951.6", "MAFF", "SLC39A12", "PCYOX1L", "CTD-3092A11.2", "RP11-271C24.3", "PRO1804", "PRR34-AS1", "SST",
    #        "CHGB", "MT1M", "JPX", "APLNR", "PPEF1"]
    col = ['RTN3', 'NAPB', 'TMSB10', 'ACTB', 'SPOCK1', 'NELL2', 'CHN1', 'TUBB2A', 'GAPDH', 'SPARCL1', 'ENO2', 'RCAN2',
           'TUBA1A', 'GABARAPL3///GABARAPL1', 'NEFM', 'ATP1B1']
    col_str = ','.join(e for e in col)

    X = df[col]
    Y = df["class"]

    clf = svm.SVC(kernel='linear')

    clf.fit(X, Y)

    scores = cross_val_score(clf, X, Y, cv=3)
    score = round(scores.mean() * 100, 2)

    file_to_write = SAMPLE_PATH / "_model.pkl"
    pickle.dump(clf, open(file_to_write, 'wb'))

    db = get_db()
    db.execute(
        "INSERT INTO modeling (user_id, trained_file, clasifier, features, model_path_name, accuracy) VALUES (?, ?, ?, ?, ?, ?)",
        (0, "GSE5281", "SVM + linear kernel", col_str, file_to_write.as_posix(), str(score)),
    )
    db.commit()

def create_user():
    #Admin of the app
    db = get_db()
    auth.create_user_db(db, "user", "user", "user", None, 1)

    db.execute("UPDATE user SET is_admin = 1 WHERE id = 1")
    db.commit()

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
    subjects = ["verify", "reset", "delete", "warning"]
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
    app.cli.add_command(init_default_model_command)
