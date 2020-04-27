import os
from flask import Blueprint, session, g, flash, url_for
from flask import render_template
from flask import request
import matplotlib.pyplot as plt
from flask import redirect

import base64
import io

from .auth import UserData, login_required
from .classes.preProcessClass import PreProcess
from .classes.featureSelectionClass import FeatureSelection

from pathlib import Path
import json

ROOT_PATH = Path.cwd()
USER_PATH = ROOT_PATH / "flaskr" / "upload" / "users"
VALIDATION_PATH = ROOT_PATH / "flaskr" / "upload" / "Validation"
GENE_INFO_PATH = ROOT_PATH / "flaskr" / "upload" / "gene_info"

bp = Blueprint("fs", __name__, url_prefix="/fs")


@bp.route("/", methods=['GET'])
@login_required
def index():

    pre_process_id = request.args.get("id")

    if pre_process_id is None:

        path = USER_PATH / str(g.user["id"])

        list_names = [f for f in os.listdir(path) if os.path.isfile((path / f))]

        if len(list_names) == 0:
            flash("Error: You don't have pre-processed or uploaded file.")

        return render_template("fs/index.html", list_names=list_names, filename=None, pre_process_id = None)

    else:

        pre_process = UserData.get_preprocess_from_id(pre_process_id)
        file_name = "GeNet_" + pre_process['file_name']

        return render_template("fs/index.html", list_names=None, filename=file_name, pre_process_id = pre_process_id)


# Get columns of 3 different feature selection methods
@bp.route("/", methods=['POST'])
@login_required
def get_val():
    fs_methods_str = request.form["fs_methods"]
    filename = request.form["change_file"]
    pre_process_id = request.form["pre_process_id"]

    user_id = session.get("user_id")

    fs_methods = fs_methods_str.split(',');

    if '' in fs_methods:
        flash("Error: Select three feature selection methods.")
        return redirect('/fs')

    file_to_open = USER_PATH / str(user_id) / filename
    df = PreProcess.getDF(file_to_open)

    if not check_df_to_fs(df):
        flash("Error: Couldn't process because having lot of features,  Start from pre-processing and reduce features.")
        return redirect(url_for('fs.index'))

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

    if pre_process_id == 'None':
        classifiers = ['4', '5', '6']
        selected_clfs_str = ','.join(e for e in classifiers)
    else:
        pre_process = UserData.get_preprocess_from_id(pre_process_id)
        selected_clfs_str = pre_process['classifiers']
        classifiers = selected_clfs_str.split(',')
        session['pre_process_id'] = None

    # Save data to the result table
    UserData.add_result(user_id, filename, fs_methods_str, selected_col[0], selected_col[1], selected_col[2], selected_clfs_str)
    result_id = UserData.get_result(user_id, filename)['id']

    results_testing, results_training = FeatureSelection.getSummaryFeatureSelection(df_m1, df_m2, df_m3, y,
                                                                                    fs_methods, classifiers)

    img64 = get_summary_plot(results_testing, results_training)

    venn_data = FeatureSelection.venn_diagram_data(col_m1, col_m2, col_m3)

    # Get gene info
    gene_info_path = GENE_INFO_PATH / "Homo_sapiens.gene_info"
    unique_genes = list(set(col_m1 + col_m2 + col_m3))
    
    gene_info_df = FeatureSelection.get_selected_gene_info(gene_info_path, unique_genes)
    gene_info = gene_info_df.to_json(orient='index')

    gene_info = json.loads(gene_info)

    gene_name_list = list(gene_info_df.index)

    return render_template("fs/result.html", image_data=img64, methods=fs_methods, columns=col, venn_data=venn_data,
                           result_id=result_id, gene_info = gene_info, gene_name_list = gene_name_list)


@bp.route("/<method>/config")
def result_config(method):
    user_id = g.user['id']
    validation_file_list = []

    if method == 'an':
        all_result = UserData.get_user_results(user_id)
        url = url_for('analyze.index')
        title = "Analysis"

    else:
        all_result = UserData.get_result_to_validation(user_id)
        url = url_for('validation.index')
        title = "Validation"

        validation_file_list = [f for f in os.listdir(VALIDATION_PATH) if os.path.isfile((VALIDATION_PATH / f))]

    all_result = [r['filename'] for r in all_result]

    return render_template("configuration.html", title=title, url=url, all_result=all_result, validation_file_list = validation_file_list)


@bp.route("/config/", methods=['POST'])
def result_config_apply():
    url = request.form["url"]
    available_result = request.form["available_result"]
    disease_file = request.form.get('disease_file')

    user_id = g.user['id']
    id = UserData.get_result(user_id, available_result)['id']

    if disease_file:
        url = url + "?id=" + str(id) + "&file=" + disease_file
    else:
        url = url + "?id=" + str(id)

    return redirect(url)

def check_df_to_fs(df):
    shape = df.shape
    #Add new things

    if shape[1] > 2000:
        return False

    else:
        return True


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
