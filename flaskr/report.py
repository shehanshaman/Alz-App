from flask import Blueprint
from flask import render_template
import pdfkit
from pathlib import Path

bp = Blueprint("report", __name__, url_prefix="/rep")

ROOT_PATH = Path.cwd()
MAIL_PATH = ROOT_PATH / "flaskr" / "mail"

@bp.route("/")
def index():
    config = pdfkit.configuration(wkhtmltopdf="C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe")
    data = render_template("report/verify_mail.html")
    pdfkit.from_string(data, 'testpdf.pdf', configuration=config)

    return render_template("report/index.html")