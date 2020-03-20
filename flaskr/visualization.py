from flask import Blueprint, session
from flask import render_template
import os
from flask import request

from flaskr.auth import login_required
from flask import g
import matplotlib.pyplot as plt

import base64

import io

from .classes.preProcessClass import PreProcess

bp = Blueprint("visualization", __name__, url_prefix="/vis")

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
USER_PATH = ROOT_PATH + "\\upload\\users\\"

@bp.route("/")
@login_required
def index():
    list_names = []
    path = USER_PATH + str(g.user["id"]) + "\\"
    if os.path.exists(path):
        for filename in os.listdir(path):
            list_names.append(filename)
        list_names.remove("tmp")

        session['files'] = list_names
        return render_template("visualization/index.html", available_list=list_names)

    return render_template("visualization/index.html")

@bp.route("/",  methods=['POST'])
def update_col():
    file_name = request.form['available_files']
    feature = ''
    if request.form.get("features", None):
        feature = request.form['features']

    if feature:
        session['select_file'] = file_name
        session['select_col'] = feature
        img64 = getPlot()
        session['columns'] = '';
        return render_template("visualization/index.html", available_list=session['files'], features_list = session['columns'], image_data=img64)

    if file_name:
        path = USER_PATH + str(g.user["id"]) + "\\" + file_name
        df = PreProcess.getDF(path)
        col = df.columns.to_list()
        session['columns'] = col

        return render_template("visualization/index.html", available_list=session['files'], features_list = session['columns'], image_data='')


def getPlot():
    file_name = session['select_file']
    feature = session['select_col']

    path = USER_PATH + str(g.user["id"]) + "\\" + file_name
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
    fig.suptitle(feature, fontsize=16)

    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    pic_hash = pic_hash.decode("utf-8")

    return pic_hash
