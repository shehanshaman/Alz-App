import os
from flask import Blueprint, session
from flask import render_template, current_app
from flask import request
import matplotlib.pyplot as plt

import base64
import io

from .auth import UserResult
from .classes.preProcessClass import PreProcess
from .classes.featureSelectionClass import FeatureSelection

bp = Blueprint("fs", __name__, url_prefix="/fs")

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
USER_PATH = ROOT_PATH + "\\upload\\users\\"

@bp.route("/")
def index():
    return render_template("fs/index.html")

#Get columns of 3 different feature selection methods
@bp.route("/" , methods=['POST'])
def get_val():
    fs_methods = request.form["fs_methods"]
    user_id = session.get("user_id")
    UserResult.update_result(user_id,'fs_methods', fs_methods)

    fs_methods = fs_methods.split(',')
    result = UserResult.get_user_results(user_id)
    filename = result['filename']
    df = PreProcess.getDF(USER_PATH + str(user_id) + "\\" + filename)

    selected_col = [None] * 3
    len = int(fs_methods[3])

    if "PCA" in fs_methods:
        i = fs_methods.index("PCA")
        pca_col = FeatureSelection.PCA(df, len).columns.tolist()
        selected_col[i] = ','.join(e for e in pca_col)


    if "Random Forest" in fs_methods:
        i = fs_methods.index("Random Forest")
        rf_col = FeatureSelection.RandomForest(df, len).columns.tolist()
        selected_col[i] = ','.join(e for e in rf_col)

    if "Extra Tree Classifier" in fs_methods:
        i = fs_methods.index("Extra Tree Classifier")
        et_col = FeatureSelection.ExtraTrees(df, len).columns.tolist()
        selected_col[i] = ','.join(e for e in et_col)

    UserResult.update_selected_col(selected_col, user_id)

    return render_template("fs/result.html")

@bp.route("/result")
def result():
    user_id = session.get("user_id")
    r = UserResult.get_user_results(user_id)
    filename = r['filename']
    df = PreProcess.getDF(USER_PATH + str(user_id) + "\\" + filename)

    # df_50F_PCA, df_50F_FI, df_50F_RF
    col = r['col_method1'].split(',')
    df_m1 = df[col]
    col = r['col_method2'].split(',')
    df_m2 = df[col]
    col = r['col_method3'].split(',')
    df_m3 = df[col]
    y = df["class"]
    method_names = r['fs_methods'].split(',')
    method_names.pop()

    results_testing, results_training = FeatureSelection.getSummaryFeatureSelection(df_m1, df_m2, df_m3,y, method_names)

    img64 = get_summary_plot(results_testing, results_training)

    return render_template("fs/result.html", image_data=img64)

def get_summary_plot(results_testing, results_training):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    # fig.subplots_adjust(wspace=0.3, hspace=0.2, top=0.8)
    # fig.set_figwidth(15)

    axes[0].set_ylim([75, 100])
    axes[1].set_ylim([75, 100])
    axes[0].set_ylabel("Accuracy %")
    axes[1].set_ylabel("Accuracy %")
    axes[0].set_title("Testing Accuracy")
    axes[1].set_title("Training Accuracy")

    fig.suptitle('Summary', fontsize=14)

    results_testing.plot.bar(rot=0, ax=axes[0])
    results_training.plot.bar(rot=0, ax=axes[1])

    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    pic_hash = pic_hash.decode("utf-8")

    return pic_hash