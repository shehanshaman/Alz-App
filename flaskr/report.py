from flask import Blueprint
from flask import render_template
import pdfkit
from pathlib import Path

import base64
import io
import matplotlib.pyplot as plt

bp = Blueprint("report", __name__, url_prefix="/rep")

ROOT_PATH = Path.cwd()
MAIL_PATH = ROOT_PATH / "flaskr" / "mail"

@bp.route("/")
def index():
    config = pdfkit.configuration(wkhtmltopdf="C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe")
    image_data = sample_img()
    data = render_template("report/verify_mail.html", image_data=image_data)
    pdfkit.from_string(data, 'testpdf.pdf', configuration=config)

    return render_template("report/index.html")

def sample_img():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot([1, 2, 3, 4])

    pic_hash = fig_to_b64encode(fig)

    return pic_hash

def fig_to_b64encode(fig):
    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    pic_hash = pic_hash.decode("utf-8")

    plt.close(fig)

    return pic_hash