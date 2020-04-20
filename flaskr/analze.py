import os
from flask import Blueprint, session, g, url_for
from flask import render_template
from flask import request
import matplotlib.pyplot as plt
import seaborn as sns
from flask import redirect

import base64
import io

from .auth import UserData, login_required
from .classes.preProcessClass import PreProcess
from .classes.featureSelectionClass import FeatureSelection

from pathlib import Path

ROOT_PATH = Path.cwd()
USER_PATH = ROOT_PATH / "flaskr" / "upload" / "users"
VALIDATION_PATH = ROOT_PATH / "flaskr" / "upload" / "Validation"

bp = Blueprint("analyze", __name__, url_prefix="/an")

@bp.route("/", methods = ['GET'])
@login_required
def index():
    result_id = request.args.get("id")

    if result_id is None:
        return redirect('../fs/an/config')

    r = UserData.get_result_from_id(result_id)
    user_id = r['user_id']

    filename = r['filename']
    file_to_open = USER_PATH / str(user_id) / filename
    df = PreProcess.getDF(file_to_open)

    col_m1 = r['col_method1'].split(',')
    col_m2 = r['col_method2'].split(',')
    col_m3 = r['col_method3'].split(',')
    method_names = r['fs_methods'].split(',')
    selected_clfs = r['classifiers'].split(',')

    len = int(method_names[3])

    col_uni = get_unique_columns(col_m1, col_m2, col_m3)

    correlation_pic_hash = get_correlation_fig(df,col_uni, method_names)

    y = df["class"]
    overlap = get_overlap_features(col_m1, col_m2, col_m3)
    overlap_str = ','.join(e for e in overlap)

    UserData.update_result_column(user_id, filename, 'col_overlapped', overlap_str)

    results, count = FeatureSelection.getFeatureSummary(df, y, col_uni, overlap, method_names, selected_clfs)
    results = results.astype(float)

    overlap_pic_hash = get_overlap_result_fig(results, count)

    # Reducing features and plot accuracies
    m1_score = FeatureSelection.returnScoreDataFrameModels(df[col_m1], y, len, selected_clfs)
    m2_score = FeatureSelection.returnScoreDataFrameModels(df[col_m2], y, len, selected_clfs)
    m3_score = FeatureSelection.returnScoreDataFrameModels(df[col_m3], y, len, selected_clfs)
    x_scores = list(map(str, FeatureSelection.get_range_array(len)))

    m_scores = [m1_score, m2_score, m3_score]

    small_set_pic_hash = get_small_set_features_fig(m_scores,x_scores, method_names[1:4], selected_clfs)

    return render_template("analyze/index.html", corr_data = correlation_pic_hash, overlap_data = overlap_pic_hash,
                           small_set= small_set_pic_hash, methods = method_names[1:4], filename=filename, result_id = result_id)


@bp.route("/step2", methods = ['POST'])
@login_required
def selected_method():

    selected_method = request.form["selected_method"]
    result_id = request.form["id"]

    r = UserData.get_result_from_id(result_id)
    user_id = r['user_id']

    filename = r['filename']
    file_to_open = USER_PATH / str(user_id) / filename
    df = PreProcess.getDF(file_to_open)
    y = df["class"]

    col_m1 = r['col_method1'].split(',')
    col_m2 = r['col_method2'].split(',')
    col_m3 = r['col_method3'].split(',')
    method_names = r['fs_methods'].split(',')
    overlap = r['col_overlapped'].split(',')
    selected_clfs = r['classifiers'].split(',')

    i = method_names.index(selected_method)

    col_uni = get_unique_columns(col_m1, col_m2, col_m3)

    cmp_corr_1, cmp_corr_2, cmp_corr_3 = get_correlation(df, col_uni)

    cmp_corr_results = FeatureSelection.compareCorrelatedFeatures(cmp_corr_1, cmp_corr_2, cmp_corr_3)
    cmp_corr_results_pic_hash = get_cmp_corr_results_fig(cmp_corr_results)

    # Corr scores and select
    corrScore = FeatureSelection.returnScoreDataFrame(df[col_uni[i]], y, selected_clfs) #0 tobe change according to the user
    corr_score_pic_hash = get_corr_score_fig(corrScore, selected_clfs)

    max_corr_df = FeatureSelection.get_max_corr_scores(corrScore, selected_clfs)

    x = df[col_uni[i]]
    col_selected_method = FeatureSelection.getSelectedDF(x, x.corr(), max_corr_df.loc[max_corr_df['Maximum Accuracy'].idxmax()]['Correlation coefficient']).columns.tolist()
    col_selected_str = ','.join(e for e in col_selected_method)

    #Update results
    UserData.update_result_column(user_id, filename, 'selected_method', selected_method)
    UserData.update_result_column(user_id, filename, 'col_selected_method', col_selected_str)

    return render_template("analyze/analyze_correlation.html", corr_results=cmp_corr_results_pic_hash,
                           tables=[max_corr_df.head().to_html(classes='data')],
                           method_title = selected_method, overlap = overlap, corr_selected = col_selected_method,
                           max_clasify = max_corr_df['Maximum Accuracy'].idxmax(), corr_score = corr_score_pic_hash,
                           filename=filename, result_id = result_id)

@bp.route("/step3", methods = ['GET'])
@login_required
def final_result():
    result_id = request.args.get("id")

    r = UserData.get_result_from_id(result_id)
    user_id = r['user_id']

    overlap = r['col_overlapped'].split(',')
    col_selected_method = r['col_selected_method'].split(',')
    selected_method = r['selected_method']

    filename = r['filename']
    file_to_open = USER_PATH / str(user_id) / filename
    df = PreProcess.getDF(file_to_open)
    y = df["class"]

    dis_gene = list(dict.fromkeys(overlap + col_selected_method))

    selected_clfs = r['classifiers'].split(',')

    r1_df = FeatureSelection.getTop3ClassificationResults_by_df(df[col_selected_method], y, selected_clfs)
    r2_df = FeatureSelection.getTop3ClassificationResults_by_df(df[dis_gene], y, selected_clfs)

    selected_roc_pic_hash = get_heatmap_roc(df[col_selected_method], y)
    all_roc_pic_hash = get_heatmap_roc(df[dis_gene], y)

    r_len = [len(col_selected_method), len(dis_gene)]
    r_col = [col_selected_method, dis_gene]

    validation_file_list = [f for f in os.listdir(VALIDATION_PATH) if os.path.isfile((VALIDATION_PATH / f))]

    return render_template("analyze/final_result.html", sel_roc=selected_roc_pic_hash, table_r1 = [r1_df.to_html(classes='data')],
                           title_r1 = r1_df.head().columns.values, all_roc=all_roc_pic_hash, table_r2 = [r2_df.to_html(classes='data')],
                           title_r2 = r2_df.head().columns.values, method = selected_method, len = r_len, col = r_col,
                           filename = filename, result_id=result_id, validation_file_list= validation_file_list)

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

    pic_hash = fig_to_b64encode(fig)

    return pic_hash

def get_overlap_result_fig(results,count):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    min_limit = int(results.min().min() / 10) * 10

    ax1.set_ylim([min_limit, 100])
    ax1.set_ylabel("Accuracy")

    results.T.plot.bar(rot=0, ax=ax1)

    ax2.set_ylim([5, 40])
    ax2.set_ylabel("No of features")
    count.plot.bar(x='id', y='val', rot=0, color=(0.2, 0.4, 0.6, 0.6), ax = ax2)
    ax2.get_legend().remove()

    pic_hash = fig_to_b64encode(fig)

    return pic_hash

def get_small_set_features_fig(m_scores, x_scores, method_names, selected_clfs):

    cls = FeatureSelection.get_cls_names(selected_clfs)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

    ax1.set_ylim(0.6, 1)
    ax1.set_xlabel('Number of genes')
    ax1.set_ylabel('Classification Accuracy')
    ax1.set_title(cls[0])

    ax1.plot(x_scores, m_scores[0][cls[0]], label='linear')
    ax1.plot(x_scores, m_scores[1][cls[0]], label='linear')
    ax1.plot(x_scores, m_scores[2][cls[0]], label='linear')

    ax1.legend(method_names, loc='lower left')

    ax2.set_ylim(0.6, 1)
    ax2.set_xlabel('Number of genes')
    ax2.set_ylabel('Classification Accuracy')
    ax2.set_title(cls[1])

    ax2.plot(x_scores, m_scores[0][cls[1]], label='linear')
    ax2.plot(x_scores, m_scores[1][cls[1]], label='linear')
    ax2.plot(x_scores, m_scores[2][cls[1]], label='linear')

    ax2.legend(method_names, loc='lower left')

    ax3.set_ylim(0.6, 1)
    ax3.set_xlabel('Number of genes')
    ax3.set_ylabel('Classification Accuracy')
    ax3.set_title(cls[2])

    ax3.plot(x_scores, m_scores[0][cls[2]], label='linear')
    ax3.plot(x_scores, m_scores[1][cls[2]], label='linear')
    ax3.plot(x_scores, m_scores[2][cls[2]], label='linear')

    ax3.legend(method_names, loc='lower left')

    pic_hash = fig_to_b64encode(fig)

    return pic_hash

def get_cmp_corr_results_fig(results):
    fig, (ax1) = plt.subplots(1, 1)

    ax1.set_xlim(0.95, 0.80)
    ax1.set_ylabel("Correlated Features %")
    ax1.set_title("Correlated Features vs correlation value")

    results.plot.line(ax=ax1)

    pic_hash = fig_to_b64encode(fig)

    return pic_hash

def get_corr_score_fig(corrScore, selected_clfs):

    cls = FeatureSelection.get_cls_names(selected_clfs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(corrScore['Correlation coefficient'], corrScore[cls[0]], label='linear')
    ax1.plot(corrScore['Correlation coefficient'], corrScore[cls[1]], label='linear')
    ax1.plot(corrScore['Correlation coefficient'], corrScore[cls[2]], label='linear')

    ax1.legend([cls[0], cls[1], cls[2], 'No of features'], loc='lower right')

    ax1.set(xlabel='correlation coefficient', ylabel='Classification Accuracy')

    ax2.plot(corrScore['Correlation coefficient'], corrScore["No of features"], label='linear')
    ax2.set(xlabel='correlation coefficient', ylabel='No of features')

    fig.subplots_adjust(wspace=0.2)
    fig.set_figwidth(15)

    pic_hash = fig_to_b64encode(fig)

    return pic_hash

def get_heatmap_roc(df, y):
    fpr, tpr, roc_auc = FeatureSelection.get_ROC_parameters(df, y)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.tight_layout(pad=10.0)

    ax1.plot(fpr[0], tpr[0], label="SVM linear, auc=" + str(round(roc_auc[0], 2)))
    ax1.plot(fpr[1], tpr[1], label="SVM gaussian, auc=" + str(round(roc_auc[1], 2)))
    ax1.plot(fpr[2], tpr[2], label="Random forest, auc=" + str(round(roc_auc[2], 2)))

    ax1.set_ylabel("Sensitivity")
    ax1.set_ylabel("1 - Specificity")

    ax1.legend(loc="lower right")

    sns.heatmap(df.corr(), cmap="RdYlGn", ax=ax2)

    pic_hash = fig_to_b64encode(fig)

    return pic_hash

def fig_to_b64encode(fig):
    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format='png')
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())

    pic_hash = pic_hash.decode("utf-8")

    return pic_hash