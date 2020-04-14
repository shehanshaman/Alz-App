from random import randint

from flask import Blueprint, g, abort, session, redirect, url_for
from flask import render_template
from flaskr.db import get_db

from flaskr.auth import login_required, UserData

bp = Blueprint("home", __name__)

@bp.route("/")
@login_required
def index():
    return render_template("home.html")
