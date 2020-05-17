from flask import session
from flask import render_template
import os

from flask import Blueprint, request
import numpy as np

from flaskr.auth import login_required
from flask import g

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import base64
import io

from .classes.preProcessClass import PreProcess

from pathlib import Path
import matplotlib.gridspec as gridspec

ROOT_PATH = Path.cwd()
USER_PATH = ROOT_PATH / "flaskr" / "upload" / "users"

bp = Blueprint("visualization", __name__, url_prefix="/vis")

@bp.route("/")
@login_required
def index():

    path = USER_PATH / str(g.user["id"])
    if os.path.exists(path):
        list_names = [f for f in os.listdir(path) if os.path.isfile((path / f))]

        session['files'] = list_names
        return render_template("visualization/index.html", available_list=list_names)

    return render_template("visualization/index.html")

@bp.route("/img/", methods=['GET'])
@login_required
def get_image_src():
    file_name = request.args.get('available_file')
    feature = request.args.get('feature').lstrip()
    img64 = getPlot(file_name, feature)

    if img64:
        return str(img64)
    else:
        return ""

@bp.route("/js/", methods=["GET"])
@login_required
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
    df = df.reset_index()

    if feature not in df.columns:
        return None

    np.warnings.filterwarnings('ignore')

    if 'class' in df.columns:

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        df.plot(kind='line', x=df.columns[0], y=df.columns[1], ax=axs[0, 0])
        axs[0, 0].get_legend().remove()
        axs[0, 0].set_ylabel('Gene Expression Values')
        axs[0, 0].set_xlabel('Sample ID')
        axs[0, 0].tick_params(labelrotation=45)
        axs[0, 0].set_title('Variation of gene expression values across samples')

        axs[0, 1].hist(df[feature])
        axs[0, 1].set_title('Histogram')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].set_xlabel('Gene Expression Values')

        df.boxplot(column=[feature], ax=axs[1, 0])
        axs[1, 0].set_title('Boxplot')
        axs[1, 0].set_ylabel('Gene Expression Values')
        axs[1, 0].set_xlabel('Gene Symbol')

        df.boxplot(column=[feature], by='class', ax=axs[1, 1])
        axs[1, 1].set_title('Boxplot group by class')
        axs[1, 1].set_ylabel('Gene Expression Values')
        axs[1, 1].set_xlabel('Different Status')

    else:
        gs = gridspec.GridSpec(2, 2)
        fig = plt.figure(figsize=(10, 8))

        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, :])

        ax1.hist(df[feature])
        ax1.set_title('Histogram')
        ax1.set_ylabel('Frequency')
        ax1.set_xlabel('Gene Expression Values')

        df.boxplot(column=[feature], ax=ax2)
        ax2.set_title('Boxplot')
        ax2.set_ylabel('Gene Expression Values')
        ax2.set_xlabel('Gene Symbol')

        df.plot(kind='line', x=df.columns[0], y=df.columns[1], ax=ax3)
        ax3.get_legend().remove()
        ax3.set_ylabel('Gene Expression Values')
        ax3.set_xlabel('Sample ID')
        ax3.tick_params(labelrotation=45)
        ax3.set_title('Variation of gene expression values across samples')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.6, wspace=0.25)

    fig.suptitle(file_name + ": " + feature, fontsize=16)

    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    pic_hash = pic_hash.decode("utf-8")

    plt.close(fig)

    return pic_hash