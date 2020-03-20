from flask import Blueprint, session
from flask import render_template
from flask import request

bp = Blueprint("fs", __name__, url_prefix="/fs")

@bp.route("/")
def index():
    return render_template("fs/index.html")

@bp.route("/" , methods=['POST'])
def get_val():
    fs_methods = request.form["fs_methods"]
    return render_template("fs/index.html")