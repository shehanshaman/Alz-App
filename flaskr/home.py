from flask import Blueprint
from flask import render_template
from flaskr.auth import login_required

bp = Blueprint("home", __name__)

@bp.route("/")
@login_required
def index():
    return render_template("home.html")