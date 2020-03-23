import os

from flask import Blueprint, session, request
from flask import render_template
from flask import redirect
from flask import g
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle

from flaskr.auth import UserResult
from flaskr.classes.preProcessClass import PreProcess

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = ROOT_PATH + "\\upload\\"
USER_PATH = UPLOAD_FOLDER + "users\\"

bp = Blueprint("modeling", __name__, url_prefix="/mod")

@bp.route("/", methods = ["GET"])
def index():
    s = 2
    if(request.method == "GET"):
        s = request.args.get('s')

    user_id = session.get("user_id")

    list_names = []
    path = USER_PATH + str(g.user["id"]) + "\\"
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path + "tmp\\")
    for filename in os.listdir(path):
        list_names.append(filename)

    list_names.remove("tmp")

    classifier_list = ["svmLinear", "svmGaussian", "randomForest"]

    r = UserResult.get_user_results(user_id)
    col_overlapped = r['col_overlapped'].split(',')
    col_selected_method = r['col_selected_method'].split(',')
    col_mo = list(dict.fromkeys(col_overlapped + col_selected_method))

    col_mo_str = ','.join(e for e in col_mo)

    UserResult.update_modeling(user_id, 'features', col_mo_str)

    return render_template("modeling/index.html", available_list=list_names, classifier_list=classifier_list,
                           features = col_mo, state = s)

@bp.route("/", methods = ["POST"])
def create_model():
    user_id = session.get("user_id")

    available_file = request.form["available_files"]
    classifier = request.form["classifier"]

    UserResult.update_modeling(user_id, 'trained_file', available_file)
    UserResult.update_modeling(user_id, 'clasifier', classifier)

    is_model = create_model_pkl(user_id, available_file, classifier)

    return redirect('/mod/?s='+ str(is_model))


@bp.route("/predict/", methods = ["GET", "POST"])
def predict():
    user_id = session.get("user_id")

    list_names = []
    path = USER_PATH + str(g.user["id"]) + "\\"
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path + "tmp\\")
    for filename in os.listdir(path):
        list_names.append(filename)

    list_names.remove("tmp")

    r = UserResult.get_user_model(user_id)
    features = r['features'].split(',')
    trained_file = r['trained_file']
    clasifier = r['clasifier']

    details = [features, trained_file, clasifier]

    if request.method == "POST":
        selected_file = request.form["available_files"]
        df = PreProcess.getDF(USER_PATH + str(user_id) + "\\" + selected_file)
        model_name = r['model_path_name']

        result = get_predicted_result_df(user_id, model_name, df[features])

    return render_template("modeling/predict.html", available_list=list_names, details = details)

def get_predicted_result_df(user_id, model_name, df):
    model = pickle.load(open(USER_PATH + str(user_id) + "\\tmp\\" + model_name, 'rb'))

    prediction = model.predict(df)

    return prediction


def create_model_pkl(user_id, filename, classifier):
    r = UserResult.get_user_model(user_id)
    col_mo = r['features'].split(',')

    df = PreProcess.getDF(USER_PATH + str(user_id) + "\\" + filename)
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
        return 0

    clf.fit(x,y)
    pickle.dump(clf, open(USER_PATH + str(user_id) + "\\tmp\\"  + "_model.pkl", 'wb'))

    UserResult.update_modeling(user_id, 'model_path_name', '_model.pkl')

    return 1

