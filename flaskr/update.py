import os
from pathlib import Path
from flask import Blueprint, current_app, request, send_from_directory, send_file, make_response

from flaskr.auth import UserResult
from flaskr.db import get_db
from .classes.preProcessClass import PreProcess

import shutil

ROOT_PATH = Path.cwd()
USER_PATH = ROOT_PATH / "flaskr" / "upload" / "users"

bp = Blueprint("update", __name__, url_prefix="/update")

@bp.route("/delete/file/", methods=["GET"])
def delete_file():
    id = request.args.get('id')
    name = request.args.get('name')

    f_path = USER_PATH / id / name

    if os.path.exists(f_path):
        os.remove(f_path)

        return '1'

    return '0'

@bp.route("/user/givenname/", methods=["GET"])
def update_given_name():
    id = request.args.get('id')
    name = request.args.get('name')

    db = get_db()
    db.execute(
        "UPDATE user SET given_name = ? WHERE id = ?",
        (name, id),
    )
    db.commit()

    return '1'

@bp.route("/user/delete/", methods=["GET"])
def delete_user_account():
    id = request.args.get('id')
    UserResult.remove_user(id)
    dir_path = USER_PATH / str(id)
    delete_folder(dir_path)
    return '1'

def delete_folder(dir_path):
    try:
        shutil.rmtree(dir_path)
        return True
    except OSError as e:
        return False

@bp.route("/download/df/", methods=["GET"])
def download_df():

    id = request.args.get('id')
    name = request.args.get('name')
    isTmp = request.args.get('is_tmp')

    if isTmp == 1:
        path = USER_PATH / str(id) / "tmp" / name
    else:
        path = USER_PATH / str(id) / name

    df = PreProcess.getDF(path)
    df_csv = df.to_csv()

    resp = make_response(df.to_csv())
    resp.headers["Content-Disposition"] = "attachment; filename=" + name.split('.')[0] + ".csv"
    resp.headers["Content-Type"] = "text/csv"

    return resp

@bp.route("/user/tour/", methods=["GET"])
def update_user_tour():

    id = request.args.get('id')
    want_tour = request.args.get('tour') # 0 & 1

    db = get_db()
    db.execute(
        "UPDATE user SET want_tour = ? WHERE id = ?",
        (want_tour, id),
    )
    db.commit()