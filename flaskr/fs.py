import os
from flask import Blueprint, session, g, flash
from flask import render_template
from flask import request
import matplotlib.pyplot as plt
from flask import redirect

import base64
import io

from flaskr.classes.validation import ValidateUser
from .auth import UserData, login_required
from .classes.preProcessClass import PreProcess
from .classes.featureSelectionClass import FeatureSelection

from pathlib import Path

ROOT_PATH = Path.cwd()
USER_PATH = ROOT_PATH / "flaskr" / "upload" / "users"

bp = Blueprint("fs", __name__, url_prefix="/fs")


@bp.route("/", methods=['GET'])
@login_required
def index():
    file_name = request.args.get("file_name")

    if file_name is None:

        list_names = []
        path = USER_PATH / str(g.user["id"])

        for filename in os.listdir(path):
            list_names.append(filename)
        list_names.remove("tmp")

        # user_id = session.get("user_id")
        # result = UserData.get_user_results(user_id)
        # filename = result['filename']
        #
        #
        # if filename is None or filename == '':
        #     flash("Error: You don't has pre-processed data-set, start from pre-processing or change file")

        return render_template("fs/index.html", list_names=list_names, filename=None)
    else:

        return render_template("fs/index.html", list_names=None, filename=file_name)


# Get columns of 3 different feature selection methods
@bp.route("/", methods=['POST'])
@login_required
def get_val():
    fs_methods_str = request.form["fs_methods"]
    filename = request.form["change_file"]

    user_id = session.get("user_id")

    fs_methods = fs_methods_str.split(',');

    if '' in fs_methods:
        flash("Error: Select three feature selection methods.")
        return redirect('/fs')

    file_to_open = USER_PATH / str(user_id) / filename
    df = PreProcess.getDF(file_to_open)

    selected_col = [None] * 3
    len = int(fs_methods[3])

    # Feature extraction from following methods
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

    # Check data already filled in database
    if UserData.get_result(user_id, filename):
        UserData.delete_result(user_id, filename)

    # calculate results
    col_m1 = selected_col[0].split(',')
    df_m1 = df[col_m1]
    col_m2 = selected_col[1].split(',')
    df_m2 = df[col_m2]
    col_m3 = selected_col[2].split(',')
    df_m3 = df[col_m3]
    y = df["class"]

    fs_methods.pop()

    col = [col_m1, col_m2, col_m3]

    if g.pre_process:
        selected_clfs_str = g.pre_process['classifiers']
        classifiers = selected_clfs_str.split(',')
        session['pre_process_id'] = None
    else:
        classifiers = ['4', '5', '6']
        selected_clfs_str = ','.join(e for e in classifiers)

    # Save data to the result table
    UserData.add_result(user_id, filename, fs_methods_str, selected_col[0], selected_col[1], selected_col[2], selected_clfs_str)
    # Save result id on session
    result_id = UserData.get_result(user_id, filename)['id']
    session['result_id'] = result_id

    results_testing, results_training = FeatureSelection.getSummaryFeatureSelection(df_m1, df_m2, df_m3, y,
                                                                                    fs_methods, classifiers)

    img64 = get_summary_plot(results_testing, results_training)

    venn_data = FeatureSelection.venn_diagram_data(col_m1, col_m2, col_m3)

    return render_template("fs/result.html", image_data=img64, methods=fs_methods, columns=col, venn_data=venn_data)


def get_summary_plot(results_testing, results_training):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    min_limit_test = int(results_testing.min().min() / 10) * 10
    min_limit_train = int(results_training.min().min() / 10) * 10

    if min_limit_test > min_limit_train:
        min_limit = min_limit_train
    else:
        min_limit = min_limit_test

    axes[0].set_ylim([min_limit, 100])
    axes[1].set_ylim([min_limit, 100])
    axes[0].set_ylabel("Accuracy %")
    axes[1].set_ylabel("Accuracy %")
    axes[0].set_title("Testing Accuracy")
    axes[1].set_title("Training Accuracy")

    fig.suptitle('Summary', fontsize=14)

    results_testing.plot.bar(rot=0, ax=axes[0])
    results_training.plot.bar(rot=0, ax=axes[1])

    pic_hash = fig_to_b64encode(fig)

    return pic_hash


def fig_to_b64encode(fig):
    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    pic_hash = pic_hash.decode("utf-8")

    return pic_hash
