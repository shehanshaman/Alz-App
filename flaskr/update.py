import os
from pathlib import Path
from flask import Blueprint, request, make_response, g, abort

from os.path import isfile, join

from flaskr.auth import UserData, login_required
from flaskr.db import get_db
from .classes.preProcessClass import PreProcess

import shutil

ROOT_PATH = Path.cwd()
USER_PATH = ROOT_PATH / "flaskr" / "upload" / "users"

bp = Blueprint("update", __name__, url_prefix="/update")


@bp.route("/delete/file/", methods=["GET"])
@login_required
def delete_file():
    id = request.args.get('id')
    name = request.args.get('name')

    UserData.delete_preprocess_file(id, name)
    UserData.delete_result(id, name)
    # UserData.delete_model(id, name)

    f_path = USER_PATH / id / name

    if os.path.exists(f_path):
        os.remove(f_path)

        return '1'

    return '0'


@bp.route("/user/givenname/", methods=["GET"])
@login_required
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
@login_required
def delete_user_account():
    id = request.args.get('id')

    #Check admin or same user
    if g.user['is_admin'] or g.user['id'] == id:

        UserData.remove_user(id)
        dir_path = USER_PATH / str(id)
        delete_folder(dir_path)

        delete_user_file(id)

        return '1'

    else:
        return abort('401')

def delete_user_file(user_id):
    files = UserData.get_user_file(user_id)

    for f in files:
        path = Path(f['path'])
        if os.path.exists( path ):
            os.remove(path)

def delete_folder(dir_path):
    try:
        shutil.rmtree(dir_path)
        return True
    except OSError as e:
        return False


@bp.route("/download/df/", methods=["GET"])
@login_required
def download_df():
    id = request.args.get('id')
    name = request.args.get('name')
    isTmp = request.args.get('is_tmp')

    if int(isTmp) == 1:
        path = USER_PATH / str(id) / "tmp" / name
        df = PreProcess.getDF(path)
    else:
        path = USER_PATH / str(id) / name
        df = PreProcess.getDF(path)
        df = df.reset_index()
        df = df.rename(columns={"index": "ID"})

    resp = make_response(df.to_csv(index=False))
    resp.headers["Content-Disposition"] = "attachment; filename=" + name.split('.')[0] + ".csv"
    resp.headers["Content-Type"] = "text/csv"

    return resp

@bp.route("/user/tour/", methods=["GET"])
@login_required
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

def is_not_admin(user):
    if user['is_admin'] == 0:
        return True
    else:
        return False

@bp.route("/user/admin/", methods=["GET"])
@login_required
def update_user_admin():
    #Check whether admin
    if is_not_admin(g.user):
        return abort('401')

    id = request.args.get('id')
    is_admin = request.args.get('is_admin')

    db = get_db()
    db.execute(
        "UPDATE user SET is_admin = ? WHERE id = ?",
        (is_admin, id),
    )
    db.commit()

    return str(is_admin)

@bp.route("/user/space/", methods=["GET"])
@login_required
def update_user_disk_space():
    # Check whether admin
    if is_not_admin(g.user):
        return abort('401')

    id = request.args.get('id')
    disk_space = request.args.get('disk_space')

    UserData.update_user_disk_space(id, disk_space)

    return "1"


@bp.route("/delete/files/", methods=["GET"])
@login_required
def delete_user_files():

    if is_not_admin(g.user):
        return abort('401')

    id = request.args.get('id')
    delete_user_all_files(id)

    return str(1)

@bp.route("/delete/tmp/", methods=["GET"])
@login_required
def delete_user_tmp_files():
    id = request.args.get('id')
    dir_path = USER_PATH / str(id) / "tmp"
    delete_files_in_dir(dir_path, True)

    return '1'

def delete_user_all_files(id):
    dir_path = USER_PATH / str(id)
    delete_files_in_dir(dir_path)
    dir_path = USER_PATH / str(id) / "tmp"
    delete_files_in_dir(dir_path)

def delete_files_in_dir(path, modal=False):
    for f in os.listdir(path):
        if modal:
            if f == '_model.pkl':
                continue
        file = join(path, f)
        if isfile(file):
            os.remove(file)

@bp.route("/warning/user/", methods=["GET"])
@login_required
def send_warning():
    id = request.args.get('id')
    UserData.send_warning(id)

    return "1"

@bp.route("/warning/users/", methods=["post"])
@login_required
def send_warnings():
    emails = request.get_json()
    UserData.send_warnings(emails)
    return "1"

@bp.route("/delete/users/", methods=["post"])
@login_required
def delete_infrequent_users():
    emails = request.get_json()
    emails_str = ','.join(('"' + e + '"') for e in emails)
    query = "SELECT * FROM user WHERE username  IN  (" + emails_str + ")"
    db = get_db()
    result = db.execute(query).fetchall()

    for r in result:
        UserData.remove_user(r['id'])
        dir_path = USER_PATH / str(r['id'])
        delete_folder(dir_path)
        delete_user_file(r['id'])

    UserData.send_delete_msg(emails)

    return "1"