import os
from flask import Blueprint, session
from flask import render_template, current_app
from flask import request
import matplotlib.pyplot as plt
import seaborn as sns

import base64
import io

from .auth import UserResult
from .classes.preProcessClass import PreProcess
from .classes.featureSelectionClass import FeatureSelection

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
USER_PATH = ROOT_PATH + "\\upload\\users\\"

bp = Blueprint("an", __name__, url_prefix="/an")

@bp.route("/")
def index():
    user_id = session.get("user_id")
    r = UserResult.get_user_results(user_id)

    filename = r['filename']
    df = PreProcess.getDF(USER_PATH + str(user_id) + "\\" + filename)

    col_m1 = r['col_method1'].split(',')
    col_m2 = r['col_method2'].split(',')
    col_m3 = r['col_method3'].split(',')
    method_names = r['fs_methods'].split(',')

    col_uni = get_unique_columns(col_m1, col_m2, col_m3)

    correlation_pic_hash = get_correlation_fig(df,col_uni, method_names)

    y = df["class"]
    overlap = get_overlap_features(col_m1, col_m2, col_m3)

    results, count = FeatureSelection.getFeatureSummary(df, y, col_uni, overlap, method_names)
    results = results.astype(float)

    overlap_pic_hash = get_overlap_result_fig(results, count)

    return render_template("analyze/index.html", corr_data = correlation_pic_hash, overlap_data = overlap_pic_hash)


def checkList(list1, list2):
    for word in list2:
        if word in list1:
            list1.remove(word)

    return list1

def get_unique_columns(col_m1,col_m2,col_m3):
    col1_uni = checkList(list(col_m1), list(col_m2 + col_m3))
    col2_uni = checkList(list(col_m2), list(col_m1 + col_m3))
    col3_uni = checkList(list(col_m3), list(col_m2 + col_m1))

    col_uni = [col1_uni, col2_uni, col3_uni]

    return col_uni

def get_overlap_features(col1, col2, col3):
    t = list(set(col1) & set(col2) & set(col3))
    return t

def get_correlation_fig(X, col, names):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    cmp_corr_1 = X[col[0]].corr()
    sns.heatmap(cmp_corr_1, cmap="RdYlGn", ax=ax1, cbar=False)
    ax1.set_title(names[0])

    cmp_corr_2 = X[col[1]].corr()
    sns.heatmap(cmp_corr_2, cmap="RdYlGn", ax=ax2, cbar=False)
    ax2.set_title(names[1])

    cmp_corr_3 = X[col[2]].corr()
    sns.heatmap(cmp_corr_3, cmap="RdYlGn", ax=ax3)
    ax3.set_title(names[2])

    fig.subplots_adjust(wspace=0.5)
    fig.set_figwidth(15)

    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    pic_hash = pic_hash.decode("utf-8")

    return pic_hash

def get_overlap_result_fig(results,count):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.set_ylim([75, 100])
    ax1.set_ylabel("Accuracy")

    results.T.plot.bar(rot=0, ax=ax1)

    ax2.set_ylim([5, 40])
    ax2.set_ylabel("No of features")
    count.plot.bar(x='id', y='val', rot=0, color=(0.2, 0.4, 0.6, 0.6), ax = ax2)
    ax2.get_legend().remove()


    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    pic_hash = pic_hash.decode("utf-8")

    return pic_hash