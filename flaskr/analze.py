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

    len = int(method_names[3])

    col_uni = get_unique_columns(col_m1, col_m2, col_m3)

    correlation_pic_hash = get_correlation_fig(df,col_uni, method_names)

    y = df["class"]
    overlap = get_overlap_features(col_m1, col_m2, col_m3)
    overlap_str = ','.join(e for e in overlap)

    UserResult.update_result(user_id, 'col_overlapped', overlap_str)

    results, count = FeatureSelection.getFeatureSummary(df, y, col_uni, overlap, method_names)
    results = results.astype(float)

    overlap_pic_hash = get_overlap_result_fig(results, count)

    # Reducing features and plot accuracies
    m1_score = FeatureSelection.returnScoreDataFrameModels(df[col_m1], y, len)
    m2_score = FeatureSelection.returnScoreDataFrameModels(df[col_m2], y, len)
    m3_score = FeatureSelection.returnScoreDataFrameModels(df[col_m3], y, len)
    x_scores = list(map(str, FeatureSelection.get_range_array(len)))

    m_scores = [m1_score, m2_score, m3_score]

    small_set_pic_hash = get_small_set_features_fig(m_scores,x_scores, method_names[1:4])

    return render_template("analyze/index.html", corr_data = correlation_pic_hash, overlap_data = overlap_pic_hash, small_set= small_set_pic_hash, methods = method_names[1:4])


@bp.route("/step2", methods=['POST'])
def selected_method():
    user_id = session.get("user_id")
    selected_method = request.form["selected_method"]
    UserResult.update_result(user_id, 'selected_method', selected_method)

    r = UserResult.get_user_results(user_id)

    filename = r['filename']
    df = PreProcess.getDF(USER_PATH + str(user_id) + "\\" + filename)
    y = df["class"]

    col_m1 = r['col_method1'].split(',')
    col_m2 = r['col_method2'].split(',')
    col_m3 = r['col_method3'].split(',')
    method_names = r['fs_methods'].split(',')

    i = method_names.index(selected_method)

    len = int(method_names[3])

    col_uni = get_unique_columns(col_m1, col_m2, col_m3)

    cmp_corr_1, cmp_corr_2, cmp_corr_3 = get_correlation(df, col_uni)

    cmp_corr_results = FeatureSelection.compareCorrelatedFeatures(cmp_corr_1, cmp_corr_2, cmp_corr_3)
    cmp_corr_results_pic_hash = get_cmp_corr_results_fig(cmp_corr_results)

    # Corr scores and select
    corrScore = FeatureSelection.returnScoreDataFrame(df[col_uni[i]], y) #0 tobe change according to the user
    cmp_corr_results_pic_hash = get_corr_score_fig(corrScore)

    max_corr_df = FeatureSelection.get_max_corr_scores(corrScore)

    x = df.drop(["class"], axis=1)
    col_selected_method = FeatureSelection.getSelectedDF(x, x.corr(), max_corr_df.loc[max_corr_df['Maximum Accuracy'].idxmax()]['i']).columns.tolist()
    col_selected_str = ','.join(e for e in col_selected_method)

    UserResult.update_result(user_id, 'col_selected_method', col_selected_str)

    return render_template("analyze/analyze_correlation.html", overlap_data=cmp_corr_results_pic_hash, tables=[max_corr_df.head().to_html(classes='data')], titles=max_corr_df.head().columns.values, method_title = selected_method)


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

def get_correlation(df, col):
    cmp_corr_1 = df[col[0]].corr()
    cmp_corr_2 = df[col[1]].corr()
    cmp_corr_3 = df[col[2]].corr()

    return cmp_corr_1, cmp_corr_2, cmp_corr_3

def get_correlation_fig(X, col, names):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    cmp_corr_1, cmp_corr_2, cmp_corr_3 = get_correlation(X, col)

    # cmp_corr_1 = X[col[0]].corr()
    sns.heatmap(cmp_corr_1, cmap="RdYlGn", ax=ax1, cbar=False)
    ax1.set_title(names[0])

    # cmp_corr_2 = X[col[1]].corr()
    sns.heatmap(cmp_corr_2, cmap="RdYlGn", ax=ax2, cbar=False)
    ax2.set_title(names[1])

    # cmp_corr_3 = X[col[2]].corr()
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

def get_small_set_features_fig(m_scores, x_scores, method_names):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

    ax1.set_ylim(0.6, 1)
    ax1.set_xlabel('Number of genes')
    ax1.set_ylabel('Classification Accuracy')
    ax1.set_title('SVM with Linear Kernal')

    ax1.plot(x_scores, m_scores[0]["svmLinear"], label='linear')
    ax1.plot(x_scores, m_scores[1]["svmLinear"], label='linear')
    ax1.plot(x_scores, m_scores[2]["svmLinear"], label='linear')

    ax1.legend(method_names, loc='lower left')

    ax2.set_ylim(0.6, 1)
    ax2.set_xlabel('Number of genes')
    ax2.set_ylabel('Classification Accuracy')
    ax2.set_title('SVM with Gaussian Kernal')

    ax2.plot(x_scores, m_scores[0]["svmGaussian"], label='linear')
    ax2.plot(x_scores, m_scores[1]["svmGaussian"], label='linear')
    ax2.plot(x_scores, m_scores[2]["svmGaussian"], label='linear')

    ax2.legend(method_names, loc='lower left')

    ax3.set_ylim(0.6, 1)
    ax3.set_xlabel('Number of genes')
    ax3.set_ylabel('Classification Accuracy')
    ax3.set_title('Random Forest')

    ax3.plot(x_scores, m_scores[0]["randomForest"], label='linear')
    ax3.plot(x_scores, m_scores[1]["randomForest"], label='linear')
    ax3.plot(x_scores, m_scores[2]["randomForest"], label='linear')

    ax3.legend(method_names, loc='lower left')

    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    pic_hash = pic_hash.decode("utf-8")

    return pic_hash

def get_cmp_corr_results_fig(results):
    fig, (ax1) = plt.subplots(1, 1)

    ax1.set_xlim(0.95, 0.80)
    ax1.set_ylabel("Correlated Features %")
    ax1.set_title("Correlated Features vs correlation value")

    results.plot.line(ax=ax1)

    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    pic_hash = pic_hash.decode("utf-8")

    return pic_hash

def get_corr_score_fig(corrScore):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(corrScore['i'], corrScore["svmLinear"], label='linear')
    ax1.plot(corrScore['i'], corrScore["svmGaussian"], label='linear')
    ax1.plot(corrScore['i'], corrScore["randomForest"], label='linear')

    ax1.legend(['svmLinear', 'svmGaussian', 'randomForest', 'No of features'], loc='lower right')

    ax1.set(xlabel='correlation coefficient', ylabel='Classification Accuracy')

    ax2.plot(corrScore['i'], corrScore["No of features"], label='linear')
    ax2.set(xlabel='correlation coefficient', ylabel='No of features')

    fig.subplots_adjust(wspace=0.2)
    fig.set_figwidth(15)

    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    pic_hash = pic_hash.decode("utf-8")

    return pic_hash