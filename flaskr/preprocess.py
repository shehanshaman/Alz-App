from flask import Blueprint, session
from flask import redirect
from flask import render_template, current_app
from flask import request

import os
from werkzeug.utils import secure_filename

from .classes.dfClass import DF #file upload instance
from .classes.preProcessClass import PreProcess
from .classes.featureReductionClass import FeatureReduction
from .classes.featureSelectionClass import FeatureSelection

import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import json
import pandas as pd

from flaskr.auth import login_required, UserResult
from flask import g

import numpy as np

bp = Blueprint("preprocess", __name__, url_prefix="/pre")

ALLOWED_EXTENSIONS = set(['pkl'])

from pathlib import Path

ROOT_PATH = Path.cwd()
USER_PATH = ROOT_PATH / "flaskr" / "upload" / "users"
UPLOAD_FOLDER = ROOT_PATH / "flaskr" / "upload"
ANNOTATION_TBL = UPLOAD_FOLDER / "AnnotationTbls"
TMP_PATH = UPLOAD_FOLDER / "tmp"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route("/")
@login_required
def index():
    list_names = []
    annotation_list = []
    path = USER_PATH / str(g.user["id"])
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path / "tmp")
    for filename in os.listdir(path):
        list_names.append(filename)

    for filename in os.listdir(ANNOTATION_TBL):
        annotation_list.append(filename)
    # print(list_names)
    list_names.remove("tmp")
    return render_template("preprocess/step-1.html", available_list=list_names, annotation_list= annotation_list)


#step 2
@bp.route("/view")
def view():
    x = json2df('user')
    if x is not None:

        if x.merge_df is None:
            df = PreProcess.mergeDF(x.path , ANNOTATION_TBL / x.anno_tbl)
            merge_name = "merge_" + x.file_name
            path = USER_PATH / str(g.user["id"]) / "tmp" / merge_name
            path_str = path.as_posix()
            PreProcess.saveDF(df, path)
            x.setMergeDF(path_str) #merge df
            df2session(x, 'user')
        else:
            df = PreProcess.getDF(x.merge_df)

        return render_template("preprocess/step-2.html", tables=[df.head(15).to_html(classes='data')], titles=df.head().columns.values)

    return redirect('/pre')

#normalization and null remove
@bp.route("/step-3", methods=['POST'])
def norm():
    x = json2df('user')

    norm_mthd = request.form["norm_mthd"]
    null_rmv = request.form["null_rmv"]

    x.setScaling(norm_mthd)
    x.setImputation(null_rmv)

    if x is not None:
        if x.merge_df is not None:
            if x.symbol_df is None:
                df = PreProcess.step3(PreProcess.getDF(x.merge_df), x.scaling, x.imputation)
                #create symbol_df
                symbol_name = "symbol_" + x.file_name
                path = USER_PATH / str(g.user["id"]) / "tmp" / symbol_name
                path_str = path.as_posix()
                PreProcess.saveDF(df, path)
                x.setSymbolDF(path_str)
                df2session(x, 'user')
            #return render_template("preprocess/index2.html", tables=[df.head().to_html(classes='data')], titles=df.head().columns.values)
            return redirect('/pre/probe2symbol')

    return redirect('/pre')

#step 3
@bp.route("/step-2")
def indexstep1():
    x = json2df('user')
    if x is not None:
        if x.merge_df is not None:

            return render_template("preprocess/step-3.html", posts="")

    return redirect('/pre')

#step 4 to 5
@bp.route("/probe2symbol")
def probe2symbol():
    x = json2df('user')
    if x is not None:
        if x.symbol_df is not None:
            if x.avg_symbol_df is None:
                df = PreProcess.probe2Symbol(PreProcess.getDF(x.symbol_df))
                avg_symbol_name = "avg_symbol_" + x.file_name
                path = USER_PATH / str(g.user["id"]) / "tmp" / avg_symbol_name
                path_str = path.as_posix()
                PreProcess.saveDF(df, path)
                x.setAvgSymbolDF(path_str)
                df2session(x, 'user')
            else:
                df = PreProcess.getDF(x.avg_symbol_df)
            return render_template("preprocess/step-4.html", tablesstep4=[df.head(15).to_html(classes='data')], titlesstep4=df.head().columns.values)

    return redirect('/pre')

#step 4
@bp.route("/step-5")
def feature_reduction():
    pvalues = np.linspace(0, 0.1, 21)
    folds = np.linspace(0, 0.1, 21)

    data_array = [pvalues, folds]

    x = json2df('user')
    p_fold_df = PreProcess.get_pvalue_fold_df(x.avg_symbol_df, x.path)
    path = USER_PATH / str(g.user["id"]) / 'tmp' / '_p_fold.pkl'
    PreProcess.saveDF(p_fold_df, path)

    return render_template("preprocess/step-5.html", data_array=data_array)

#step 6
@bp.route("/step-6/", methods=['POST'])
def get_reduce_features_from_pvalues():
    x = json2df('user')

    user_tmp_path = USER_PATH / str(g.user["id"]) / 'tmp'
    path = user_tmp_path / '_p_fold.pkl'
    p_fold_df = PreProcess.getDF(path)

    fold= request.form["fold-range"]
    pvalue = request.form["p-value"]
    df = PreProcess.get_filtered_df_pvalue(p_fold_df, x.avg_symbol_df, float(pvalue), float(fold))
    fr_df_path = user_tmp_path / ('fr_' + x.file_name)
    PreProcess.saveDF(df, fr_df_path)
    x.setReduceDF(fr_df_path.as_posix())
    df2session(x, 'user')

    length = len(df.columns)

    if(length < 350 ):
        split_array = np.linspace(150, int(length / 10) * 10, 21)
    else:
        split_array = np.linspace(150, 350, 21)

    split_array = split_array.astype(int)

    return render_template("preprocess/step-6.html", split_array = split_array)


@bp.route("/fr/pf/", methods=['GET'])
def get_feature_count_pval():
    x = json2df('user')
    pvalue = request.args.get("pvalue")
    foldChange = request.args.get("foldChange")

    path = USER_PATH / str(g.user["id"]) / 'tmp' / '_p_fold.pkl'
    p_fold_df = PreProcess.getDF(path)

    count = PreProcess.get_filtered_df_count_pvalue(p_fold_df, float(pvalue), float(foldChange))
    return str(count)

@bp.route("/fr/save/", methods=['POST'])
def save_reduced_df():
    features_count = request.form['features_count']
    x = json2df('user')
    df = PreProcess.getDF(x.reduce_df)
    df_y = PreProcess.getDF(x.path)
    y = df_y['class']
    df_selected = FeatureReduction.getSelectedFeatures(df, int(features_count), y)
    path = USER_PATH / str(g.user["id"]) / ('re_' + x.file_name)
    PreProcess.saveDF(df_selected, path)
    user_id = session.get("user_id")
    UserResult.update_result(user_id, 'filename', 're_' + x.file_name)

    return redirect('/fs/')

@bp.route("/fr2" , methods=['POST'])
def FR_selected():
    if request.method == 'POST':
        features_count = request.form['features_count']
        file_to_open = UPLOAD_FOLDER / "other" / "GSE5281_DE_2311.plk"
        df = PreProcess.getDF(file_to_open)
        df_200 = FeatureReduction.getSelectedFeatures(df, int(features_count))

        #testing only
        # df_obj_tmp = DF("a", "b", "c")
        x = json2df('user')
        path = TMP_PATH + "reduce_" + x.file_name
        PreProcess.saveDF(df_200, path)
        x.setReduceDF(path)
        df2session(x, 'user')

        return redirect('/pre/fs')

    return redirect('/')

#step 7
@bp.route("/fs")
def FS():
    return render_template("preprocess/fs.html", posts="")


@bp.route('/', methods=['POST'])
def create_object():
    if request.method == 'POST':
        anno_tbl = request.form["anno_tbl"]
        column_selection = request.form["column_selection"]
        available_file = request.form["available_files"]

        path = USER_PATH / str(g.user["id"])

        if anno_tbl and column_selection and available_file:
            df_obj = DF(file_name=available_file, path=os.path.join(path, available_file), anno_tbl=anno_tbl,
                        col_sel_method=column_selection, merge_df=None,
                        symbol_df=None, avg_symbol_df=None, reduce_df=None, scaling=None, imputation=None)
            json_data = json.dumps(df_obj.__dict__)
            session['user'] = json_data
            return redirect('/pre/view')

    return redirect('/pre/')

@bp.route('/upload')
def upload_file_view():
    return render_template("preprocess/step-0.html")

# file upload
@bp.route('/upload/', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        file = request.files['chooseFile']

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            path = USER_PATH / str(g.user["id"]) / filename
            file.save(path)
            return redirect('/pre')
        else:
            return redirect(request.url)

@bp.route('/plot_fr.png')
def plot_png():
    df = PreProcess.getDF(UPLOAD_FOLDER + "\\other\\GSE5281_DE_2311.plk")
    selectedFeatures = FeatureReduction.getScoresFromUS(df)
    fig = FeatureReduction.create_figure(selectedFeatures)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def json2df(df_name):
    if session.get(df_name):
        json_data = session[df_name]
        df = DF( ** json.loads(json_data))
        return df

    return None


def df2session(obj, name):
    json_data = json.dumps(obj.__dict__)
    session[name] = json_data
