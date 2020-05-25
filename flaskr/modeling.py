import os

import pandas as pd
from flask import Blueprint, request, url_for
from flask import render_template
from flask import redirect
from flask import g
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import pickle
import json

from flaskr.auth import UserData, login_required
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
    result_id = request.args.get("id")
    if result_id:
        result = UserData.get_result_from_id(result_id)
        analysed_file = result['filename']
    else:
        analysed_file = None

    s = -1
    if request.method == "GET":
        s = request.args.get('s')
        a = request.args.get('a')

    user_id = g.user['id']

    classifier_list = ["svmLinear", "svmGaussian", "randomForest"]

    all_result = UserData.get_result_to_validation(user_id)
    all_result = [r['filename'] for r in all_result]

    return render_template("modeling/index.html", available_list='', classifier_list=classifier_list,
                           state=s, accuracy=a, all_result=all_result, analysed_file = analysed_file)

@bp.route("/files/", methods=["GET"])
@login_required
def get_files_for_modeling():
    filename = request.args.get('filename')

    user_id = g.user['id']

    result = UserData.get_result_for_modeling(user_id, filename)

    col_overlapped = result['col_overlapped'].split(',')
    col_selected_method = result['col_selected_method'].split(',')

    col = list(set(col_overlapped + col_selected_method))

    path = USER_PATH / str(user_id)
    list_names = []

    for f in os.listdir(path):
        file_path = path / f
        if os.path.isfile(file_path):
            df = PreProcess.getDF(file_path)
            if ValidateUser.is_subset(df.columns.to_list(), col):
                list_names.append(f)

    return json.dumps(list_names)


@bp.route("/", methods=["POST"])
@login_required
def create_model():
    user_id = g.user['id']

    available_file = request.form["available_files"]
    classifier = request.form["classifier"]
    available_result_file = request.form["available_result"]

    result = UserData.get_result(user_id, available_result_file)

    score = create_model_pkl(user_id, available_file, classifier, result)

    if score is None:
        return redirect('/mod/?s=0')
    else:
        return redirect('/mod/?s=1&a=' + str(score))


@bp.route("/predict/")
@login_required
def predict():
    user_id = g.user['id']

    path = USER_PATH / str(g.user["id"])

    list_names = [f for f in os.listdir(path) if os.path.isfile((path / f))]

    annotation_list = []
    annotation_db = UserData.get_annotation_file(g.user["id"])
    for f in annotation_db:
        annotation_list.append([f['file_name'], f['path']])

    r = UserData.get_model(user_id)
    default_r = UserData.get_model(0)

    if r['accuracy']:
        # return redirect(url_for('modeling.index') + "?s=2")

        features = r['features'].split(',')
        trained_file = r['trained_file']
        clasifier = r['clasifier']
        accuracy = r['accuracy']
        accuracy = str(round(float(accuracy), 2))

        details = [features, trained_file, clasifier, accuracy]

    else:
        details = None

    default_features = default_r['features'].split(',')
    default_trained_file = default_r['trained_file']
    default_clasifier = default_r['clasifier']
    default_accuracy = default_r['accuracy']
    default_accuracy = str(round(float(default_accuracy), 2))


    default_details = [default_features, default_trained_file, default_clasifier, default_accuracy]

    return render_template("modeling/predict.html", available_list=list_names, details=details,
                           annotation_list=annotation_list, default_details= default_details)


# add new
@bp.route("/predict/results/", methods=["POST"])
@login_required
def predict_results():
    user_id = g.user['id']
    is_default_model = request.form.get("is_default_model")
    if int(is_default_model):
        r = UserData.get_model(0)
    else:
        r = UserData.get_model(user_id)

    features = r['features'].split(',')

    selected_file = request.form["available_files"]
    df_path = USER_PATH / str(user_id) / selected_file
    df = PreProcess.getDF(df_path)

    is_norm = request.form.get("is_norm")
    is_map = request.form.get("is_map")

    if is_map == "true":
        annotation_file = request.form["anno_tbl"]
        annotation_table_path = UPLOAD_FOLDER.as_posix() + annotation_file
        df = PreProcess.mergeDF(df_path, Path(annotation_table_path) )
        df = df.dropna(axis=0, subset=['Gene Symbol'])
        df = PreProcess.probe2Symbol(df)
        df = PreProcess.step3(df, 'sklearn', 'drop')
        df = df.set_index(['Gene Symbol'])
        df = df.T

    elif is_norm == "true":
        df = get_norm_df(df)

    model_name = r['model_path_name']

    e = ValidateUser.has_col(df.columns, features)
    if e is not None:
        return render_template("error.html", errors=e)

    result = get_predicted_result_df(user_id, model_name, df[features], is_default_model)
    result = result.astype(str)
    result[result == '0'] = 'Negative'
    result[result == '1'] = 'Positive'

    frame = {'ID': df.index, 'Predicted Result': result}
    out_result = pd.DataFrame(frame)

    save_path = USER_PATH / str(user_id) / "tmp" / "results.pkl"
    out_result.to_pickle(save_path);

    data = out_result['Predicted Result'].value_counts()

    return render_template("modeling/predict-results.html",
                           tables=[out_result.to_html(classes='display" id = "table_id')], data=data)


@bp.route("/results/", methods=["GET"])
@login_required
def get_results_for_modeling():
    filename = request.args.get('filename')
    user_id = g.user['id']

    result = UserData.get_result(user_id, filename)
    result = '|'.join(str(r) for r in result)

    return result


def get_predicted_result_df(user_id, model_name, df, is_default_model):
    if int(is_default_model):
        model = pickle.load(open(Path(model_name), 'rb'))
    else:
        model = pickle.load(open(USER_PATH / str(user_id) / "tmp" / model_name, 'rb'))

    prediction = model.predict(df)

    return prediction


def create_model_pkl(user_id, filename, classifier, result):
    file_to_open = USER_PATH / str(user_id) / filename
    df = PreProcess.getDF(file_to_open)

    col_overlapped = result['col_overlapped'].split(',')
    col_selected_method = result['col_selected_method'].split(',')
    col_mo = list(dict.fromkeys(col_overlapped + col_selected_method))

    col_mo_str = ','.join(e for e in col_mo)

    e = ValidateUser.has_col(df.columns, col_mo)
    if e is not None:
        return None

    y = df["class"]
    x = df[col_mo]

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
    score = round(scores.mean() * 100, 2)

    file_to_write = USER_PATH / str(user_id) / "tmp" / "_model.pkl"
    pickle.dump(clf, open(file_to_write, 'wb'))

    UserData.update_model(user_id, filename, classifier, col_mo_str, "_model.pkl", str(score))

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
