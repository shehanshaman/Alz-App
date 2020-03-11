from flask import Blueprint
from flask import flash
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

bp = Blueprint("preprocess", __name__, url_prefix="/pre")

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'pkl'])

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = ROOT_PATH + "\\upload\\"
ANNOTATION_TBL = UPLOAD_FOLDER + "AnnotationTbls\\GPL570-55999.csv"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route("/")
def index():
    print(current_app.config['APP_ALZ'].df)
    return render_template("preprocess/step-1.html", posts="")
    #return render_template("preprocess/index2.html", posts="")

#first step table show
@bp.route("/view")
def view():

    if(current_app.config['APP_ALZ'].df != ''):
        df = PreProcess.mergeDF(current_app.config['APP_ALZ'].df.path , ANNOTATION_TBL)
        # df = PreProcess.getDF(current_app.config['APP_ALZ'].df.path)
        current_app.config['APP_ALZ'].df.setMergeDF(df) #merge df

        return render_template("preprocess/step-2.html", tables=[df.head().to_html(classes='data')], titles=df.head().columns.values)

    else:
        return redirect('/pre')

#normalization and null remove
@bp.route("/step-3", methods=['POST'])
def norm():
    x = current_app.config['APP_ALZ'].df
    if(x != ''):
        df = PreProcess.step3(x.merge_df)
        x.setSymbolDF(df)
        #return render_template("preprocess/index2.html", tables=[df.head().to_html(classes='data')], titles=df.head().columns.values)
        return redirect('/pre/probe2symbol')
    else:
        return redirect('/pre')

#step 2
@bp.route("/step-2")
def indexstep1():
    return render_template("preprocess/step-3.html", posts="")

#step 4 to 5
@bp.route("/probe2symbol")
def probe2symbol():
    x = current_app.config['APP_ALZ'].df
    if(x != ''):
        df = PreProcess.probe2Symbol(x.symbol_df)
        x.setAvgSymbolDF(df)
        return render_template("preprocess/step-4.html", tablesstep4=[df.head().to_html(classes='data')], titlesstep4=df.head().columns.values)
    else:
        return redirect('/pre')

#step 4
@bp.route("/step-5")
def indexstep2():
    return render_template("preprocess/step-5.html", posts="")

#step 6
@bp.route("/fr")
def FR():
    # print(df_200.shape)
    return render_template("preprocess/feRe.html", posts="")


@bp.route("/fr" , methods=['POST'])
def FR_selected():
    if request.method == 'POST':
        features_count = request.form['features_count']
        df = PreProcess.getDF(UPLOAD_FOLDER + "\\other\\GSE5281_DE_2311.plk")
        df_200 = FeatureReduction.getSelectedFeatures(df, int(features_count))

        #testing only
        df_obj_tmp = DF("a", "b", "c")
        df_obj_tmp.setReduceDF(df_200)
        current_app.config['APP_ALZ'].df = df_obj_tmp

        return redirect('/pre/fs')

    return redirect('/')

#step 7
@bp.route("/fs")
def FS():
    return render_template("preprocess/fs.html", posts="")

@bp.route("/fs" , methods=['POST'])
def FS_post():
    x = current_app.config['APP_ALZ'].df

    if request.method == 'POST':
        features_count = request.form['features_count']
        df_pca = FeatureSelection.PCA(x.reduce_df, int(features_count))
        df_rf = FeatureSelection.RandomForest(x.reduce_df, int(features_count))
        df_et = FeatureSelection.ExtraTrees(x.reduce_df, int(features_count))
        return render_template("preprocess/tableView.html", tables=[df_et.head().to_html(classes='data')],
                               titles=df_et.head().columns.values)

    return render_template("preprocess/fs.html", posts="")

#file upload
@bp.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        anno_tbl = request.form["anno_tbl"]
        # check if the post request has the file part
        if 'chooseFile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['chooseFile']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            anno_tbl = request.form["anno_tbl"]
            column_selection = request.form["column_selection"]
            filename = secure_filename(file.filename)
            df_obj = DF(os.path.join(UPLOAD_FOLDER, filename), anno_tbl, column_selection)
            current_app.config['APP_ALZ'].df = df_obj
            file.save(df_obj.path)
            flash('File successfully uploaded')
            return redirect('/pre/view')
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)

@bp.route('/plot_fr.png')
def plot_png():
    df = PreProcess.getDF(UPLOAD_FOLDER + "\\other\\GSE5281_DE_2311.plk")
    selectedFeatures = FeatureReduction.getScoresFromUS(df)
    fig = FeatureReduction.create_figure(selectedFeatures)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
