import functools
import os

from datetime import datetime

import pandas as pd
from flask_mail import Message

from flask import Blueprint, current_app
from flask import flash
from flask import g
from flask import redirect
from flask import render_template
from flask import request
from flask import session
from flask import url_for
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash

from flaskr.db import get_db
import random
import string
from pathlib import Path

import requests
import json

ROOT_PATH = Path.cwd()
USER_PATH = ROOT_PATH / "flaskr" / "upload" / "users"

bp = Blueprint("auth", __name__, url_prefix="/auth")


def login_required(view):
    """View decorator that redirects anonymous users to the login page."""

    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for("auth.login"))

        return view(**kwargs)

    return wrapped_view


@bp.before_app_request
def load_logged_in_user():
    """If a user id is stored in the session, load the user object from
    the database into ``g.user``."""
    user_id = session.get("user_id")

    if user_id is None:
        g.user = None
        g.dList = None
    else:
        g.dList = UserData.get_disable_validate_array(user_id)
        g.user = (
            get_db().execute("SELECT * FROM user WHERE id = ?", (user_id,)).fetchone()
        )

@bp.route("/register", methods=("GET", "POST"))
def register():
    """Register a new user.

    Validates that the username is not already taken. Hashes the
    password for security.
    """
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        given_name = request.form["given_name"]

        db = get_db()
        error = None

        if not username:
            error = "Username is required."
        elif not password:
            error = "Password is required."
        elif (
            db.execute("SELECT id FROM user WHERE username = ?", (username,)).fetchone()
            is not None
        ):
            error = "User {0} is already registered.".format(username)

        if error is None:
            # the name is available, store it in the database and go to
            # the login page

            create_user_db(db, username, password, given_name, '', 1)
            if "@" in username:
                user_id = UserData.get_user_id(username)
                verify_key = randomString()
                db.execute(
                    "INSERT INTO verify (user_id, subject, verify_key) VALUES (?, ?, ?)",
                    (user_id, 'verify', verify_key),
                )
                db.commit()
                url = "http://" + str(request.host) + "/auth/verify/?id=" + str(user_id) + "&key=" + verify_key
                send_mail("verify", url, username, "Verify email address")

            message  = given_name + ", Your account created. Verify your email"
            flash(message)
            return redirect(url_for("auth.login"))

        flash(error)

    return render_template("auth/register.html")


@bp.route("/login", methods=("GET", "POST"))
def login():
    """Log in a registered user by adding the user id to the session."""
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        db = get_db()
        error = None
        user = db.execute(
            "SELECT * FROM user WHERE username = ?", (username,)
        ).fetchone()

        if user is None:
            error = "Incorrect username."
        elif not check_password_hash(user["password"], password):
            error = "Incorrect password."
        elif user["is_verified"] == 0:
            error = "Your account not verified, check email."

        if error is None:
            # store the user id in a new session and return to the index
            session.clear()
            session["user_id"] = user["id"]
            update_last_login(db, user["id"])
            return redirect(url_for("index"))

        flash(error)

    return render_template("auth/login.html")


@bp.route("/logout")
def logout():
    """Clear the current session, including the stored user id."""
    session.clear()
    return redirect(url_for("index"))

@bp.route("/glogin/", methods=["POST"])
def glogin():
    email = request.form["email"]
    given_name = request.form["given_name"]
    profile_id = request.form["profile_id"]
    image_url = request.form["image_url"]

    db = get_db()
    user = db.execute(
        "SELECT * FROM user WHERE username = ?", (email,)
    ).fetchone()

    if user is None:
        # Register User
        create_user_db(db, email, profile_id, given_name, image_url, 2)
        user = db.execute(
            "SELECT * FROM user WHERE username = ?", (email,)
        ).fetchone()

    else:
        # Update user last login
        update_last_login(db, user["id"])

    session.clear()
    session["user_id"] = user["id"]

    return redirect(url_for("index"))


@bp.route("/verify/", methods=["GET"])
def verify():
    user_id = request.args.get('id')
    verify_key = request.args.get('key')

    db = get_db()
    verify_data = db.execute(
        "SELECT * FROM verify WHERE user_id = ? AND subject = 'verify'", (user_id,)
    ).fetchone()

    e = ["Not Found",[]]

    if verify_data is None:
        e[1].append("Not registered user.")

    elif verify_key == verify_data['verify_key']:
        db.execute(
            "UPDATE user SET is_verified = ? WHERE id = ?",
            (1, user_id),
        )
        db.commit()

        db.execute(
            "DELETE FROM verify WHERE user_id = ? AND subject = 'verify'",
            (user_id, ),
        )
        db.commit()
        flash("Your email has been verified.")
        return redirect(url_for("auth.login"))

    else:
        e[1].append("Wrong Key.")

    return render_template("error.html", errors=e)

@bp.route("/reset", methods = ["POST", "GET"])
def reset_request():

    if request.method == "POST":
        username = request.form["username"]
        db = get_db()
        user = db.execute(
            "SELECT * FROM user WHERE username = ? AND is_verified = 1", (username,)
        ).fetchone()

        if user is None:
            flash("Wrong username.")
            return render_template("auth/reset_request.html")

        user_id = user["id"]

        verify_key = randomString()

        db.execute(
            "INSERT INTO verify (user_id, subject, verify_key) VALUES (?, ?, ?)",
            (user_id, 'reset', verify_key),
        )
        db.commit()
        url = "http://" + str(request.host) + "/auth/reset/?id=" + str(user_id) + "&key=" + verify_key
        send_mail("reset", url, username, "Reset Password Alz-App")

        message = "Please, check your email."
        flash(message)

        return redirect(url_for('auth.login'))

    return render_template("auth/reset_request.html")


@bp.route("/reset/", methods = ["GET", "POST"])
def reset_key_verify():

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        #Update password
        db = get_db()
        db.execute(
            "UPDATE user SET password = ? WHERE username = ?",
            (generate_password_hash(password), username),
        )
        db.commit()

        #Get user id
        user = db.execute(
            "SELECT * FROM user WHERE username = ?", (username,)
        ).fetchone()

        user_id = int(user['id'])

        #Delete query in verify
        db.execute(
            "DELETE FROM verify WHERE user_id = ? AND subject = 'reset'",
            (user_id,)
        )
        db.commit()

        flash("Your password has been reset.")
        return redirect(url_for("auth.login"))


    user_id = request.args.get('id')
    verify_key = request.args.get('key')

    db = get_db()
    verify_data = db.execute(
        "SELECT * FROM verify WHERE user_id = ? AND subject = 'reset' ORDER BY id DESC", (user_id,)
    ).fetchone()

    if verify_data is None:
        flash("You don't request for reset.")
        return redirect(url_for("auth.login"))

    elif verify_key == verify_data['verify_key']:

        flash("Your email has been verified, Enter new password.")

        user = db.execute(
            "SELECT * FROM user WHERE id = ?", (user_id,)
        ).fetchone()

        return render_template("auth/reset.html", email = user["username"])

    else:
        flash("Wrong url for rest verification.")
        return redirect(url_for("auth.login"))


@bp.route("/settings")
@login_required
def settings():
    user = g.user
    user_data = [user['username'], user['given_name']]

    path = USER_PATH / str(user['id'])
    df_files = get_files_size(path)

    folder_size = round(sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / 1024 / 1024, 2)
    max_usage = current_app.config['APP_ALZ'].max_usage

    full_usage = folder_size
    file_usage = round(df_files['file size'].sum(), 2)
    cache_usage = round((full_usage - file_usage) ,2)
    available_space = round((max_usage - full_usage) , 2)

    if available_space < 0:
        available_space = 0

    usages = [file_usage, cache_usage, available_space]

    data = [user_data, usages]

    return render_template("auth/settings.html", data = data, df_files = df_files)

def get_all_users():
    db = get_db()
    users = db.execute(
        "SELECT * FROM user",
    ).fetchall()

    col = ["id", "username", "given_name", "last_login", "is_verified", "is_admin"]

    df = pd.DataFrame(columns=col)

    for user in users:
        df2 = pd.DataFrame([[user['id'], user['username'], user['given_name'], user['last_login'], user['is_verified'], user['is_admin']]], columns=col)
        df = df.append(df2)

    return df

def get_folder_size(users_id):
    col = ["id", "usage"]
    df = pd.DataFrame(columns=col)

    for user_id in users_id:
        root_directory = USER_PATH / str(user_id)
        folder_size = round(sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file()) / 1024 / 1024 , 2)
        df2 = pd.DataFrame([[user_id, folder_size]], columns=col)
        df = df.append(df2, ignore_index=True)

    return df

@bp.route("/admin")
@login_required
def admin_panel():

    users = get_all_users()
    host_usage = get_folder_size(users["id"].tolist())

    users = pd.merge(users, host_usage, on='id')

    infrequent_ids = get_infrequent_ids(users)
    select_users = users[(users['id'].isin(infrequent_ids))]
    ids_str = ','.join(str(e) for e in select_users['id'].tolist())

    data = [round(select_users['usage'].sum(), 2) , select_users.shape[0], ids_str]

    return render_template("auth/admin.html", select_users=select_users, users=users, data=data)

#contact list show
@bp.route("/admin/contact_list")
@login_required
def admin_contact_panel():
    contact_data = {'option':'ok', 'password':'abc123', 'option_name':'contact_list'}
    try:
        res = requests.post('http://localhost/myapp/fyp-web-app/web-app/data/data_extract_api.php', data=contact_data)

    except requests.exceptions.RequestException as e:
        flash("Requested host not available")
        return redirect(url_for('auth.admin_panel'))

    contact_list = res.text
    
    return render_template("auth/contact_list.html", contact_list=contact_list)

#subscribe list show
@bp.route("/admin/subscribe_list")
@login_required
def admin_subscribe_panel():
    contact_data = {'option':'ok', 'password':'abc123', 'option_name':'subscribe_list'}
    try:
        res = requests.post('http://localhost/myapp/fyp-web-app/web-app/data/data_extract_api.php', data=contact_data)

    except requests.exceptions.RequestException as e:
        flash("Requested host not available")
        return redirect(url_for('auth.admin_panel'))

    subscribe_list = res.text
    
    return render_template("auth/subscribe_list.html", subscribe_list=subscribe_list)

def get_infrequent_ids(users):
    n = datetime.now()
    list = []
    for index, row in users.iterrows():
        u_log = datetime.strptime(row['last_login'], '%Y-%m-%d %H:%M:%S.%f')
        delta = n - u_log
        if delta.days > 1 and row['usage'] > 10:
            list.append(row['id'])

    return list


def get_files_size(path):

    file_names = []
    file_sizes = []

    files = [f for f in os.listdir(path)]
    for file in files:
        filename = os.path.join(path, file)
        if os.path.isfile(filename):
            file_names.append(file)
            file_size = os.path.getsize(filename) / 1024 / 1024
            file_sizes.append(file_size)

    df = pd.DataFrame(data={"file name": file_names, "file size": file_sizes}, columns=["file name", "file size"])
    folder_size = df['file size'].sum()
    df['percentage'] = round(df['file size'] * 100 / folder_size, 2)

    return df

def send_mail(subject, url, recipient, senders_subject):
    msg = Message(senders_subject,
                  sender="no-reply@GeNet.com",
                  recipients=[recipient])

    message = get_mail_message(subject)
    message = message.replace("{{action_url}}", url)
    msg.html = message
    mail = current_app.config["APP_ALZ"].mail
    s = mail.send(msg)

    return s

def send_infrequent_mail(recipients):
    msg = Message("Unavailable uploaded files",
                  sender="no-reply@alz.com",
                  recipients=recipients)
    message = get_mail_message("infrequent")
    msg.html = message
    mail = current_app.config["APP_ALZ"].mail
    s = mail.send(msg)

    return s

def get_mail_message(subject):
    db = get_db()
    m = db.execute(
        "SELECT message FROM mail_template WHERE subject = ?",
        (subject,),
    ).fetchone()
    return m['message']

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def create_user_db(db, username, password, given_name, image_url, is_verified):
    db.execute(
        "INSERT INTO user (username, password, given_name, image_url, last_login, is_verified) VALUES (?, ?, ?, ?, ?,?)",
        (username, generate_password_hash(password), given_name, image_url, datetime.now(), is_verified),
    )
    db.commit()

    user_id = UserData.get_user_id(username)
    db.execute(
        "INSERT INTO modeling (user_id, trained_file) VALUES (?, ?)",
        (user_id, None),
    )
    db.commit()

    path = USER_PATH / str(user_id)
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path / "tmp")

    return True

def update_last_login(db, user_id):
    db.execute(
        "UPDATE user SET last_login = ? WHERE id = ?",
        (datetime.now(), user_id),
    )
    db.commit()



class UserData:

    def get_user_id(username):
        db = get_db()
        user = db.execute(
            "SELECT * FROM user WHERE username = ?", (username,)
        ).fetchone()
        if user is not None:
            return user["id"]
        return None

    def get_user_all_preprocess(user_id):
        db = get_db()
        result = db.execute(
            "SELECT * FROM preprocess WHERE user_id = ?", (user_id,)
        ).fetchall()
        return result

    def get_user_preprocess(user_id, file_name):
        db = get_db()
        result = db.execute(
            "SELECT * FROM preprocess WHERE user_id = ? AND file_name = ?", (user_id, file_name)
        ).fetchone()
        return result

    def get_preprocess_from_id(id):
        db = get_db()
        result = db.execute(
            "SELECT * FROM preprocess WHERE id = ?", (id, )
        ).fetchone()
        return result

    def update_preprocess(user_id, file_name, column, value):
        db = get_db()
        db.execute(
            "UPDATE preprocess SET " + column + " = ? WHERE user_id = ? AND file_name = ?", (value, user_id, file_name),
        )
        db.commit()

    def add_preprocess(user_id, file_name, file_path, annotation_table, col_sel_method, merge_df_path):
        db = get_db()
        db.execute(
            "INSERT INTO preprocess (user_id, file_name, file_path, annotation_table, col_sel_method, merge_df_path) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, file_name, file_path, annotation_table, col_sel_method, merge_df_path),
        )
        db.commit()

    def delete_preprocess_file(user_id, file_name):
        db = get_db()
        db.execute(
            "DELETE FROM preprocess WHERE user_id = ? AND file_name = ?",
            (user_id, file_name),
        )
        db.commit()

    def delete_preprocess_all_file(user_id):
        db = get_db()
        db.execute(
            "DELETE FROM preprocess WHERE user_id = ?",
            (user_id),
        )
        db.commit()

    #result Table
    def get_result(user_id, filename):
        db = get_db()
        result = db.execute(
            "SELECT * FROM results WHERE user_id = ? AND filename = ?", (user_id, filename)
        ).fetchone()
        return result

    def get_result_from_id(id):
        db = get_db()
        result = db.execute(
            "SELECT * FROM results WHERE id = ?", (id,)
        ).fetchone()
        return result

    def get_result_to_validation(user_id):
        db = get_db()
        result = db.execute(
            "SELECT * FROM results WHERE user_id = ? AND selected_method != '' ", (user_id,)
        ).fetchall()

        return result

    def get_result_for_modeling(user_id, filename):
        db = get_db()
        result = db.execute(
            "SELECT * FROM results WHERE user_id = ? AND selected_method != '' AND filename = ?", (user_id, filename),
        ).fetchone()

        return result

    def add_result(user_id, filename, fs_methods, col_method1, col_method2, col_method3, classifiers):
        db = get_db()
        db.execute(
            "INSERT INTO results (user_id, filename, fs_methods, col_method1, col_method2, col_method3, classifiers) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, filename, fs_methods, col_method1, col_method2, col_method3, classifiers),
        )
        db.commit()

    def delete_result(user_id, filename):
        db = get_db()
        db.execute(
            "DELETE FROM results WHERE user_id = ? AND filename = ?",
            (user_id, filename),
        )
        db.commit()

    def delete_results(user_id):
        db = get_db()
        db.execute(
            "DELETE FROM results WHERE user_id = ?",
            (user_id,),
        )
        db.commit()

    def update_result_column(user_id, filename, column, value):
        db = get_db()
        db.execute(
            "UPDATE results SET " + column +" = ? WHERE user_id = ? AND filename = ?",( value, user_id, filename),
        )
        db.commit()

    def get_user_results(user_id):
        db = get_db()
        result = db.execute(
            "SELECT * FROM results WHERE user_id = ?", (user_id,)
        ).fetchall()

        return result

    #Medeling Table
    def update_model(user_id, trained_file, clasifier, features, model_path_name, accuracy):
        db = get_db()
        db.execute(
            "UPDATE modeling SET trained_file = ?, clasifier = ?, features = ?, model_path_name = ?, "
            "accuracy = ?  WHERE user_id = ?", (trained_file, clasifier, features, model_path_name, accuracy, user_id),
        )
        db.commit()

    def get_model(user_id):
        db = get_db()
        result = db.execute(
            "SELECT * FROM modeling WHERE user_id = ?", (user_id,)
        ).fetchone()
        return result

    #remove user from the app
    def remove_user(user_id):
        db = get_db()
        db.execute(
            "DELETE FROM user WHERE id = ?",
            (user_id),
        )
        db.commit()

        db.execute(
            "DELETE FROM modeling WHERE user_id = ?",
            (user_id),
        )
        db.commit()

        UserData.delete_results(user_id)
        UserData.delete_preprocess_all_file(user_id)

    def infrequent_users(ids):
        db = get_db()
        query = "SELECT * FROM user WHERE id IN (" + ids + ")"
        result = db.execute(query).fetchall()

        list = []
        for user in result:
            x =user['username']
            list.append(x)

        send_infrequent_mail(list)

        return list

    def is_file_upload(user_id):
        path = USER_PATH / str(user_id)

        if os.path.exists(path):
            list_names = [f for f in os.listdir(path) if os.path.isfile((path / f))]

            if len(list_names) == 0:
                return False
            else:
                return True

        return False

    def delete_model(user_id, name):
        db = get_db()
        db.execute(
            "UPDATE modeling SET "
            "accuracy = ?  WHERE user_id = ? and trained_file = ?", ('', user_id, name),
        )
        db.commit()

    def is_model_created(user_id):
        user_model = UserData.get_model(user_id)
        if user_model and user_model['accuracy']:
            return True

        return False

    def get_disable_validate_array(user_id):
        disable_list = [0, 0, 0, 0, 0, 0, 0]

        if(UserData.is_file_upload(user_id)):
            disable_list[0] = 1
            disable_list[1] = 1
            disable_list[2] = 1
        if(UserData.get_user_results(user_id)):
            disable_list[3] = 1
        if(UserData.get_result_to_validation(user_id)):
            disable_list[4] = 1
            disable_list[5] = 1
        if UserData.is_model_created(user_id) and disable_list[0]:
            disable_list[6] = 1  
        return disable_list

    def add_file(file_name, file_type, path, user_id, is_annotation, has_class):
        db = get_db()
        db.execute(
            "INSERT INTO file (file_name, file_type, path, user_id, is_annotation, has_class) VALUES (?, ?, ?, ?, ?, ?)",
            (file_name, file_type, path, user_id, is_annotation, has_class),
        )
        db.commit()

    def get_annotation_file(user_id):
        db = get_db()
        result = db.execute(
            "SELECT * FROM file WHERE user_id IN (0, ?) AND is_annotation = 1", (user_id,)
        ).fetchall()
        return result

    def get_user_file(user_id):
        db = get_db()
        result = db.execute(
            "SELECT * FROM file WHERE user_id = ?", (user_id,)
        ).fetchall()
        return result

    def get_user_file_by_file_name(user_id, file_name):
        db = get_db()
        result = db.execute(
            "SELECT * FROM file WHERE user_id = ? AND file_name = ?", (user_id, file_name),
        ).fetchone()
        return result

    def delete_user_file(user_id):
        db = get_db()
        db.execute(
            "DELETE FROM file WHERE user_id = ?",
            (user_id),
        )
        db.commit()

    def delete_user_file_by_file_name(user_id, file_name):
        db = get_db()
        db.execute(
            "DELETE FROM file WHERE user_id = ? AND file_name = ?",
            (user_id, file_name),
        )
        db.commit()