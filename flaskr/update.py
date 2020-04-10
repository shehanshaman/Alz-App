import os
from pathlib import Path
from flask import Blueprint, current_app, request, send_from_directory, send_file, make_response

from os.path import isfile, join

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
    s = 1
    id = request.args.get('id')
    want_tour = request.args.get('tour')  # 0 & 1

    if want_tour == 'false':
        s = 0

    db = get_db()
    db.execute(
        "UPDATE user SET want_tour = ? WHERE id = ?",
        (s, id),
    )
    db.commit()

    return str(want_tour)


@bp.route("/user/admin/", methods=["GET"])
def update_user_admin():
    id = request.args.get('id')
    is_admin = request.args.get('is_admin')

    db = get_db()
    db.execute(
        "UPDATE user SET is_admin = ? WHERE id = ?",
        (is_admin, id),
    )
    db.commit()

    return str(is_admin)


@bp.route("/delete/files/", methods=["GET"])
def delete_user_files():
    id = request.args.get('id')
    delete_user_all_files(id)

    return str(1)

def delete_user_all_files(id):
    dir_path = USER_PATH / str(id)
    delete_files_in_dir(dir_path)
    dir_path = USER_PATH / str(id) / "tmp"
    delete_files_in_dir(dir_path)

def delete_files_in_dir(path):
    for f in os.listdir(path):
        file = join(path, f)
        if isfile(file):
            os.remove(file)

@bp.route("/infrequent/files/", methods=["GET"])
def infrequent_files_delete():
    ids = request.args.get('ids')
    id_array = ids.split(',')

    for id in id_array:
        delete_user_all_files(id)

    UserResult.infrequent_users(ids)

    return "1"

@bp.route("/infrequent/ntfy/", methods=["GET"])
def infrequent_user_ntfy():
    ids = request.args.get('ids')
    id_array = ids.split(',')

    return "1"