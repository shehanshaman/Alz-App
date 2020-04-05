import os

import pandas as pd
from flask import Blueprint, session, request
from flask import render_template
from flask import redirect
from flask import g
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import pickle

from flaskr.auth import UserResult, login_required
from flaskr.classes.preProcessClass import PreProcess

from pathlib import Path

from flaskr.classes.validation import *

ROOT_PATH = Path.cwd()
USER_PATH = ROOT_PATH / "flaskr" / "upload" / "users"
UPLOAD_FOLDER = ROOT_PATH / "flaskr" / "upload"
ANNOTATION_TBL = UPLOAD_FOLDER / "AnnotationTbls"

bp = Blueprint("modeling", __name__, url_prefix="/mod")


@bp.route("/", methods=["GET"])
@login_required
def index():
    s = 2
    if request.method == "GET":
        s = request.args.get('s')
        a = request.args.get('a')

    user_id = session.get("user_id")

    list_names = []
    path = USER_PATH / str(g.user["id"])
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path / "tmp")
    for filename in os.listdir(path):
        list_names.append(filename)

    list_names.remove("tmp")

    classifier_list = ["svmLinear", "svmGaussian", "randomForest"]

    r = UserResult.get_user_results(user_id)

    e = ValidateUser.has_data(r, ['col_overlapped', 'col_selected_method'])

    if e is not None:
        return render_template("error.html", errors=e)

    col_overlapped = r['col_overlapped'].split(',')
    col_selected_method = r['col_selected_method'].split(',')
    col_mo = list(dict.fromkeys(col_overlapped + col_selected_method))

    col_mo_str = ','.join(e for e in col_mo)

    UserResult.update_modeling(user_id, 'features', col_mo_str)

    return render_template("modeling/index.html", available_list=list_names, classifier_list=classifier_list,
                           features=col_mo, state=s, accuracy = a)


@bp.route("/", methods=["POST"])
@login_required
def create_model():
    user_id = session.get("user_id")

    available_file = request.form["available_files"]
    classifier = request.form["classifier"]

    UserResult.update_modeling(user_id, 'trained_file', available_file)
    UserResult.update_modeling(user_id, 'clasifier', classifier)

    score = create_model_pkl(user_id, available_file, classifier)

    if score is None:
        return redirect('/mod/?s=0')
    else:
        return redirect('/mod/?s=1&a='+str(score * 100))



@bp.route("/predict/", methods=["GET", "POST"])
@login_required
def predict():
    user_id = session.get("user_id")

    list_names = []
    path = USER_PATH / str(g.user["id"])
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path / "tmp")
    for filename in os.listdir(path):
        list_names.append(filename)

    list_names.remove("tmp")

    annotation_list = []
    for filename in os.listdir(ANNOTATION_TBL):
        annotation_list.append(filename)

    r = UserResult.get_user_model(user_id)

    e = ValidateUser.has_data(r, ['features', 'trained_file', 'clasifier', 'model_path_name'])

    if e is not None:
        return render_template("error.html", errors = e)

    features = r['features'].split(',')
    trained_file = r['trained_file']
    clasifier = r['clasifier']
    accuracy = r['accuracy']

    details = [features, trained_file, clasifier, accuracy]

    if request.method == "POST":
        selected_file = request.form["available_files"]
        df_path = USER_PATH / str(user_id) / selected_file
        df = PreProcess.getDF(df_path)

        is_norm = request.form.get("is_norm")
        is_map = request.form.get("is_map")

        if is_map == "true":
            annotation_file = request.form["anno_tbl"]
            df = PreProcess.mergeDF(df_path, ANNOTATION_TBL / annotation_file)
            df = PreProcess.step3(df, 'sklearn', 'drop')
            df = PreProcess.probe2Symbol(df)
            df = df.set_index(['Gene Symbol'])
            df = df.T

        elif is_norm == "true":
            df = get_norm_df(df)

        model_name = r['model_path_name']

        e = ValidateUser.has_col(df.columns, features)
        if e is not None:
            return render_template("error.html", errors=e)

        result = get_predicted_result_df(user_id, model_name, df[features])
        result = result.astype(str)
        result[result == '0'] = 'Negative'
        result[result == '1'] = 'Positive'

        frame = {'ID': df.index, 'Predicted Result': result}
        out_result = pd.DataFrame(frame)

        return render_template("modeling/predict.html", available_list=list_names, details=details,
                               annotation_list=annotation_list,
                               tables=[out_result.to_html(classes='display" id = "table_id')])

    return render_template("modeling/predict.html", available_list=list_names, details=details,
                           annotation_list=annotation_list, tables='')

#add new
@bp.route("/predict/results/", methods=["GET", "POST"])
@login_required
def predict_results():
    user_id = session.get("user_id")

    list_names = []
    path = USER_PATH / str(g.user["id"])
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path / "tmp")
    for filename in os.listdir(path):
        list_names.append(filename)

    list_names.remove("tmp")

    annotation_list = []
    for filename in os.listdir(ANNOTATION_TBL):
        annotation_list.append(filename)

    r = UserResult.get_user_model(user_id)

    e = ValidateUser.has_data(r, ['features', 'trained_file', 'clasifier', 'model_path_name'])

    if e is not None:
        return render_template("error.html", errors = e)

    features = r['features'].split(',')
    trained_file = r['trained_file']
    clasifier = r['clasifier']
    accuracy = r['accuracy']

    details = [features, trained_file, clasifier, accuracy]

    if request.method == "POST":
        selected_file = request.form["available_files"]
        df_path = USER_PATH / str(user_id) / selected_file
        df = PreProcess.getDF(df_path)

        is_norm = request.form.get("is_norm")
        is_map = request.form.get("is_map")

        if is_map == "true":
            annotation_file = request.form["anno_tbl"]
            df = PreProcess.mergeDF(df_path, ANNOTATION_TBL / annotation_file)
            df = PreProcess.step3(df, 'sklearn', 'drop')
            df = PreProcess.probe2Symbol(df)
            df = df.set_index(['Gene Symbol'])
            df = df.T

        elif is_norm == "true":
            df = get_norm_df(df)

        model_name = r['model_path_name']

        e = ValidateUser.has_col(df.columns, features)
        if e is not None:
            return render_template("error.html", errors=e)

        result = get_predicted_result_df(user_id, model_name, df[features])
        result = result.astype(str)
        result[result == '0'] = 'Negative'
        result[result == '1'] = 'Positive'

        frame = {'ID': df.index, 'Predicted Result': result}
        out_result = pd.DataFrame(frame)

        save_path = USER_PATH / str(user_id) / "results.pkl";
        #out_result.to_csv(save_path, index = False)
        out_result.to_pickle(save_path);
        
        return render_template("modeling/predict-results.html", available_list=list_names, details=details,
                               annotation_list=annotation_list,
                               tables=[out_result.to_html(classes='display" id = "table_id')])

    return render_template("modeling/predict-results.html", available_list=list_names, details=details,
                           annotation_list=annotation_list, tables='')
#add new

def get_predicted_result_df(user_id, model_name, df):
    model = pickle.load(open(USER_PATH / str(user_id) / "tmp" / model_name, 'rb'))

    prediction = model.predict(df)

    return prediction


def create_model_pkl(user_id, filename, classifier):
    r = UserResult.get_user_model(user_id)
    col_mo = r['features'].split(',')
    file_to_open = USER_PATH / str(user_id) / filename
    df = PreProcess.getDF(file_to_open)

    e = ValidateUser.has_col(df.columns, col_mo)
    if e is not None:
        return None

    y = df["class"]
    x = df[col_mo]

    clf = ''

    if classifier == "svmLinear":
        clf = svm.SVC(kernel='linear')
    elif classifier == "svmGaussian":
        clf = SVC(kernel="rbf", gamma="auto", C=1)
    elif classifier == "randomForest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
    else:
        return None

    clf.fit(x, y)

    scores = cross_val_score(clf, x, y, cv=3)
    score = round(scores.mean(), 2)
    UserResult.update_modeling(user_id, 'accuracy', str(score))

    file_to_write = USER_PATH / str(user_id) / "tmp" / "_model.pkl"
    pickle.dump(clf, open(file_to_write , 'wb'))

    UserResult.update_modeling(user_id, 'model_path_name', '_model.pkl')

    return score


def get_norm_df(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_new = pd.DataFrame(data=x_scaled, columns=df.columns)
    df_new.insert(loc=0, column=df.columns.name, value=df.index)
    df_new = df_new.set_index([df.columns.name])
    df_new.index.name = None
    return df_new
