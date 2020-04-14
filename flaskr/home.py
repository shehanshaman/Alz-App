from random import randint

from flask import Blueprint, g, abort, session, redirect, url_for
from flask import render_template
from flaskr.db import get_db

from flaskr.auth import login_required, UserData

bp = Blueprint("home", __name__)

@bp.route("/")
@login_required
def index():
    id = g.user['id']
    pre = UserData.get_user_pre_request(id)

    db = get_db()
    query = "SELECT * FROM prerequest"
    result = db.execute(query).fetchall()

    list = []
    for u in result:
        x = u['prerequest']
        list.append(x)

    # abort(405)

    return render_template("home.html", dis = pre, pre_name = list)

@bp.route("/<session_name>/reset")
@login_required
def reset_session(session_name):
    session[session_name] = None

    if session_name == "pre_process_id":
        url = url_for('preprocess.index')
    else:
        url = url_for('fs.index')

    return redirect(url)