from flask import Blueprint, session
from flask import render_template
import os
from flask import request

from flask import send_file
from flask import Blueprint, current_app, request, send_from_directory, send_file, make_response

from flaskr.auth import login_required
from flask import g
import matplotlib.pyplot as plt

import base64

import io

from .classes.preProcessClass import PreProcess

from pathlib import Path

ROOT_PATH = Path.cwd()
USER_PATH = ROOT_PATH / "flaskr" / "upload" / "users"

bp = Blueprint("visualization", __name__, url_prefix="/vis")

@bp.route("/")
@login_required
def index():
    list_names = []
    path = USER_PATH / str(g.user["id"])
    if os.path.exists(path):
        for filename in os.listdir(path):
            list_names.append(filename)
        list_names.remove("tmp")

        session['files'] = list_names
        return render_template("visualization/index.html", available_list=list_names)

    return render_template("visualization/index.html")


@bp.route("/", methods=['POST'])
@login_required
def update_col():
    file_name = request.form['available_files']
    feature = request.form['features']
    img64 = getPlot(file_name, feature)

    return render_template("visualization/index.html", available_list=session['files'], image_data=img64)


@bp.route("/js/", methods=["GET"])
def get_col_names_js():
    file_name = request.args.get('available_files')
    user_id = request.args.get('user_id')
    path = USER_PATH / str(user_id) / file_name
    df = PreProcess.getDF(path)
    col = df.columns.tolist()
    col_str = ','.join(e for e in col)

    return col_str


def getPlot(file_name, feature):
    path = USER_PATH / str(g.user["id"]) / file_name
    df = PreProcess.getDF(path)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].scatter(df['class'], df[feature], edgecolors='r')
    axs[0, 0].set_title('Scatter plot')
    axs[0, 1].hist(df[feature])
    axs[0, 1].set_title('Histogram')
    df.boxplot(column=[feature], ax=axs[1, 0])
    axs[1, 0].set_title('Boxplot')
    df.boxplot(column=[feature], by='class', ax=axs[1, 1])
    axs[1, 1].set_title('Boxplot group by class')
    fig.suptitle(file_name + ": " + feature, fontsize=16)

    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    pic_hash = pic_hash.decode("utf-8")

    save_path = USER_PATH / str(g.user["id"]) / "visualization.pdf";
    fig.savefig(save_path)
    return pic_hash

@bp.route("/download/img/", methods=["GET"])
def download_df():

    id = request.args.get('id')
    name = request.args.get('name')

    path = USER_PATH / str(id) / name

    return send_file(path, attachment_filename='visualization.pdf')

# def encode(data: bytes):
#     """
#     Return base-64 encoded value of binary data.
#     """
#     return base64.b64encode(data)


# def decode(data: str):
#     """
#     Return decoded value of a base-64 encoded string.
#     """
#     return base64.b64decode(data.encode())


# def get_pdf_data(filename):
#     """
#     Open pdf file in binary mode,
#     return a string encoded in base-64.
#     """
#     with open(filename, 'rb') as file:
#         return encode(file.read())

# @bp.route("/download/img/", methods=["GET"])
# def download_df():

#     id = request.args.get('id')
#     name = request.args.get('name')

#     path = USER_PATH / str(id) / name

#     # df = PreProcess.getDF(path)
#     # df_csv = df.to_csv()
#     data_pdf = get_pdf_data(path)
#     #data_decode = decode(data_pdf)

#     resp = make_response(data_pdf)
#     resp.headers["Content-Disposition"] = "attachment; filename=" + name.split('.')[0] + ".pdf"
#     resp.headers["Content-Type"] = "application/pdf"

#     return resp

