from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from flaskr.db import get_db


class FeatureSelection:

    def PCA(df_200F, len):
        y = df_200F["class"]
        X = df_200F.drop("class",1).values
        initial_feature_names = df_200F.columns.values

        # 160 samples with 2312 features
        train_features = X

        model = PCA(n_components=100).fit(train_features)  # n_components=63
        X_pc = model.transform(train_features)

        # number of components
        n_pcs = model.components_.shape[0]

        # get the index of the most important feature on EACH component i.e. largest absolute value
        # using LIST COMPREHENSION HERE
        most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

        # get the names
        most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

        df_50F_PCA = df_200F[most_important_names]
        df_50F_PCA = df_50F_PCA.T.drop_duplicates().T
        # df_50F_PCA.head()

        # apply SelectKBest class to extract top 10 best features
        bestfeatures = SelectKBest(score_func=chi2, k=10)
        fit = bestfeatures.fit(df_50F_PCA, y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(df_50F_PCA.columns)

        # concat two dataframes for better visualization
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
        selectedFeatures = featureScores.nlargest(len, 'Score')

        pd.options.mode.chained_assignment = None

        df_50F_PCA = df_200F[selectedFeatures['Specs'].values]
        # df_50F_PCA.head()

        return df_50F_PCA

    def RandomForest(df_200F, len):
        y = df_200F["class"]
        X = df_200F.drop("class", 1)

        model = RandomForestClassifier(n_estimators=10000, random_state=42, n_jobs=-1)
        model.fit(X, y)

        feat_importances = pd.Series(model.feature_importances_, index=X.columns)

        rf_selected = feat_importances.nlargest(len)
        df_50F_RF = df_200F[pd.DataFrame(rf_selected).T.columns]
        # df_50F_RF.head()
        return df_50F_RF

    def ExtraTrees(df_200F, len):
        y = df_200F["class"]
        X = df_200F.drop("class", 1)

        model = ExtraTreesClassifier()
        model.fit(X, y)

        feat_importances = pd.Series(model.feature_importances_, index=X.columns)

        fi_selected = feat_importances.nlargest(len)
        df_50F_FI = df_200F[pd.DataFrame(fi_selected).T.columns]
        # df_50F_FI.head()

        return df_50F_FI

    def getTop3ClassificationResults(X_train, X_test, y_train, y_test, selected_clfs):

        clf = [None] * 3
        score = [None] * 3

        if "1" in selected_clfs:
            i = selected_clfs.index("1")
            gnb_clf = GaussianNB()
            score_gnb = cross_val_score(gnb_clf, X_train, y_train, cv=3)
            gnb_clf.fit(X_train, y_train)
            clf[i] = gnb_clf
            score[i] = score_gnb

        if "2" in selected_clfs:
            i = selected_clfs.index("2")
            dt_clf = DecisionTreeClassifier()
            score_dt = cross_val_score(dt_clf, X_train, y_train, cv=3)
            dt_clf.fit(X_train, y_train)
            clf[i] = dt_clf
            score[i] = score_dt

        if "3" in selected_clfs:
            i = selected_clfs.index("3")
            knn_clf = KNeighborsClassifier(n_neighbors=2)
            score_knn = cross_val_score(knn_clf, X_train, y_train, cv=3)
            knn_clf.fit(X_train, y_train)
            clf[i] = knn_clf
            score[i] = score_knn

        if "4" in selected_clfs:
            i = selected_clfs.index("4")
            svm_rbf_clf = SVC(kernel="rbf", gamma="auto", C=1)
            score_svm_rbf = cross_val_score(svm_rbf_clf, X_train, y_train, cv=3)
            svm_rbf_clf.fit(X_train, y_train)
            clf[i] = svm_rbf_clf
            score[i] = score_svm_rbf

        if "5" in selected_clfs:
            i = selected_clfs.index("5")
            svm_li_clf = svm.SVC(kernel='linear')  # Linear Kernel
            score_svm_li = cross_val_score(svm_li_clf, X_train, y_train, cv=3)
            svm_li_clf.fit(X_train, y_train)
            clf[i] = svm_li_clf
            score[i] = score_svm_li

        if "6" in selected_clfs:
            i = selected_clfs.index("6")
            RF_clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
            score_RF = cross_val_score(RF_clf, X_train, y_train, cv=3)
            RF_clf.fit(X_train, y_train)
            clf[i] = RF_clf
            score[i] = score_RF

        y_pred = [None] * 3
        y_pred[0] = clf[0].predict(X_test)
        y_pred[1] = clf[1].predict(X_test)
        y_pred[2] = clf[2].predict(X_test)

        m = np.array([[round(metrics.accuracy_score(np.int64(y_test.values), y_pred[0]) * 100, 2),
                       round(score[0].mean() * 100, 2)],
                      [round(metrics.accuracy_score(np.int64(y_test.values), y_pred[1]) * 100, 2),
                       round(score[1].mean() * 100, 2)],
                      [round(metrics.accuracy_score(np.int64(y_test.values), y_pred[2]) * 100, 2),
                       round(score[2].mean() * 100, 2)]])

        return m

    def getTop3ClassificationResults_by_df(df, y, selected_clfs):
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)
        m = FeatureSelection.getTop3ClassificationResults(X_train, X_test, y_train, y_test, selected_clfs)
        classifier_result_df = pd.DataFrame(m, columns=["Testing", "Training"])
        classifiers = FeatureSelection.get_cls_names(selected_clfs)
        classifier_result_df["index"] = classifiers
        classifier_result_df = classifier_result_df.set_index('index')
        classifier_result_df.index.name = None

        return classifier_result_df

    def get_cls_names(selected_clfs):
        db = get_db()
        cls = db.execute(
            "SELECT * FROM classifiers WHERE id IN (?,?,?)", (selected_clfs[0], selected_clfs[1], selected_clfs[2])
        ).fetchall()


        cls_name = [c['clf_name'] for c in cls]

        return cls_name

    def getSummaryFeatureSelection(df_50F_PCA, df_50F_FI, df_50F_RF, y, rows, selected_clfs):

        cla = FeatureSelection.get_cls_names(selected_clfs)

        X_train, X_test, y_train, y_test = train_test_split(df_50F_PCA, y, test_size=0.3, random_state=42)
        m_pca = FeatureSelection.getTop3ClassificationResults(X_train, X_test, y_train, y_test, selected_clfs)

        X_train, X_test, y_train, y_test = train_test_split(df_50F_FI, y, test_size=0.3, random_state=42)
        m_fi = FeatureSelection.getTop3ClassificationResults(X_train, X_test, y_train, y_test, selected_clfs)

        X_train, X_test, y_train, y_test = train_test_split(df_50F_RF, y, test_size=0.3, random_state=42)
        m_rf = FeatureSelection.getTop3ClassificationResults(X_train, X_test, y_train, y_test, selected_clfs)

        # rows = ["classifier", "PCA", "Random Forest", "Extra Tree"]
        rows.insert(0, "classifier")
        # cla = ["SVM + Gaussian kernel", "SVM + Linear kernel", "Random forest"]

        data_testing = np.array([cla, m_pca[:, 0], m_fi[:, 0], m_rf[:, 0]])
        results_testing = pd.DataFrame(data=data_testing, index=rows).transpose()

        data_training = np.array([cla, m_pca[:, 1], m_fi[:, 1], m_rf[:, 1]])
        results_training = pd.DataFrame(data=data_training, index=rows).transpose()

        results_testing = results_testing.set_index('classifier')
        results_training = results_training.set_index('classifier')

        results_testing = results_testing.astype(float)
        results_training = results_training.astype(float)

        return results_testing, results_training

    def getFeatureSummary(df, y, col_uni, overlap, rows, selected_clfs):

        list0 = FeatureSelection.get_cls_names(selected_clfs)

        count = [len(col_uni[0]), len(col_uni[1]), len(col_uni[2]), len(overlap)]

        diff_df = df[col_uni[0]]

        X_train, X_test, y_train, y_test = train_test_split(diff_df, y, test_size=0.3, random_state=42)
        list1 = FeatureSelection.getTop3ClassificationResults(X_train, X_test, y_train, y_test, selected_clfs)[:, 0]  # same dataset

        diff_df = df[col_uni[1]]

        X_train, X_test, y_train, y_test = train_test_split(diff_df, y, test_size=0.3, random_state=42)
        list2 = FeatureSelection.getTop3ClassificationResults(X_train, X_test, y_train, y_test, selected_clfs)[:, 0]  # same dataset

        diff_df = df[col_uni[2]]

        X_train, X_test, y_train, y_test = train_test_split(diff_df, y, test_size=0.3, random_state=42)
        list3 = FeatureSelection.getTop3ClassificationResults(X_train, X_test, y_train, y_test, selected_clfs)[:, 0]  # same dataset

        diff_df = df[overlap]

        X_train, X_test, y_train, y_test = train_test_split(diff_df, y, test_size=0.3, random_state=42)
        list4 = FeatureSelection.getTop3ClassificationResults(X_train, X_test, y_train, y_test, selected_clfs)[:, 0]  # same dataset

        rows.insert(0, "classifier")
        rows.insert(4, "Overlap")
        rows.pop()
        # rows = ["classifier", "PCA", "Random Forest", "Extra Tree", "Overlap"]

        data = np.array([list0, list1, list2, list3, list4])
        results = pd.DataFrame(data=data, index=rows).transpose()

        results = results.set_index('classifier')

        df_count = pd.DataFrame({'id': rows[1:5], 'val': count})

        return results, df_count



    def returnScoreDataFrameModels(dataFrame, y, len, selected_clfs):

        lists = [[], [], []]

        for i in FeatureSelection.get_range_array(len):

            if "1" in selected_clfs:
                j = selected_clfs.index("1")
                lists[j].append(FeatureSelection.gaussianNB(dataFrame.iloc[:, 0:(i)], y))

            if "2" in selected_clfs:
                j = selected_clfs.index("2")
                lists[j].append(FeatureSelection.decisionTree(dataFrame.iloc[:, 0:(i)], y))

            if "3" in selected_clfs:
                j = selected_clfs.index("3")
                lists[j].append(FeatureSelection.kNeighbors(dataFrame.iloc[:, 0:(i)], y))

            if "4" in selected_clfs:
                j = selected_clfs.index("4")
                lists[j].append(FeatureSelection.svmGaussian(dataFrame.iloc[:, 0:(i)], y))

            if "5" in selected_clfs:
                j = selected_clfs.index("5")
                lists[j].append(FeatureSelection.svmLinear(dataFrame.iloc[:, 0:(i)], y))

            if "6" in selected_clfs:
                j = selected_clfs.index("6")
                lists[j].append(FeatureSelection.randomForest(dataFrame.iloc[:, 0:(i)], y))

        rows = FeatureSelection.get_cls_names(selected_clfs)

        modelScore = pd.DataFrame(data=lists, index=rows).transpose()

        return modelScore

    def get_range_array(len):
        array = [100, 90, 80, 70, 60, 50, 40, 30, 20, 15, 10, 5, 3, 2, 1]
        return array[array.index(FeatureSelection.roundupten(len)):]

    def gaussianNB(dataFrame, target):
        clf = GaussianNB()
        scores = cross_val_score(clf, dataFrame, target, cv=3)

        return scores.mean()

    def decisionTree(dataFrame, target):
        clf = DecisionTreeClassifier()
        scores = cross_val_score(clf, dataFrame, target, cv=3)

        return scores.mean()

    def kNeighbors(dataFrame, target):
        clf = KNeighborsClassifier(n_neighbors=2)
        scores = cross_val_score(clf, dataFrame, target, cv=3)

        return scores.mean()

    def svmLinear(dataFrame, target):
        clf = svm.SVC(kernel='linear')  # Linear Kernel
        scores = cross_val_score(clf, dataFrame, target, cv=3)

        return scores.mean()

    def svmGaussian(dataFrame, target):
        # Create a svm Classifier
        clf = SVC(kernel="rbf", gamma="auto", C=1)
        scores = cross_val_score(clf, dataFrame, target, cv=3)

        return scores.mean()

    def randomForest(dataFrame, target):
        # Create a svm Classifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
        scores = cross_val_score(clf, dataFrame, target, cv=3)

        return scores.mean()

    def roundupten(x):
        return int(x / 10.0) * 10

    def getHighlyCorrelatedFeatures(corr, i):
        # Select upper triangle of correlation matrix
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > i)]

        return to_drop

    def compareCorrelatedFeatures(cmp_corr_pca, cmp_corr_fi, cmp_corr_rf):
        lists0 = []
        lists1 = []
        lists2 = []
        lists3 = []

        i = 0.95

        while i >= 0.8:
            lists0.append(i)
            lists1.append(len(FeatureSelection.getHighlyCorrelatedFeatures(cmp_corr_pca, i)) * 100 / 34)
            lists2.append(len(FeatureSelection.getHighlyCorrelatedFeatures(cmp_corr_rf, i)) * 100 / 18)
            lists3.append(len(FeatureSelection.getHighlyCorrelatedFeatures(cmp_corr_fi, i)) * 100 / 18)

            i = i - 0.005

        rows = ["Correlation value", "PCA", "Random Forest", "Extra Tree"]

        data = np.array([lists0, lists1, lists2, lists3])
        results = pd.DataFrame(data=data, index=rows).transpose()
        results = results.set_index('Correlation value')

        return results

    def getSelectedDF(df, corr, ran):
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[0]):
                if corr.iloc[i, j] >= ran or corr.iloc[i, j] <= -ran:
                    if columns[j]:
                        columns[j] = False

        selected_columns = df.columns[columns]
        df_selected = df[selected_columns]

        return df_selected

    def returnScoreDataFrame(dataFrame, y, selected_clfs):

        lists = [[], [], [], [], []]

        i = 1

        while i >= 0.6:

            df_tmp = dataFrame
            df_tmp = FeatureSelection.getSelectedDF(df_tmp, df_tmp.corr(), i)

            lists[0].append(i)
            lists[4].append(len(df_tmp.columns))

            if "1" in selected_clfs:
                j = selected_clfs.index("1") + 1
                lists[j].append(FeatureSelection.gaussianNB(df_tmp, y))

            if "2" in selected_clfs:
                j = selected_clfs.index("2") + 1
                lists[j].append(FeatureSelection.decisionTree(df_tmp, y))

            if "3" in selected_clfs:
                j = selected_clfs.index("3") + 1
                lists[j].append(FeatureSelection.kNeighbors(df_tmp, y))

            if "4" in selected_clfs:
                j = selected_clfs.index("4") + 1
                lists[j].append(FeatureSelection.svmGaussian(df_tmp, y))

            if "5" in selected_clfs:
                j = selected_clfs.index("5") + 1
                lists[j].append(FeatureSelection.svmLinear(df_tmp, y))

            if "6" in selected_clfs:
                j = selected_clfs.index("6") + 1
                lists[j].append(FeatureSelection.randomForest(df_tmp, y))

            # i = i - 0.0025
            i = i - 0.01

        rows = FeatureSelection.get_cls_names(selected_clfs)
        rows.insert(0, "Correlation coefficient")
        rows.insert(4, "No of features")

        results = pd.DataFrame(data=lists, index=rows).transpose()

        return results

    def get_max_corr_scores(corrScore, selected_clfs):

        cls = FeatureSelection.get_cls_names(selected_clfs)

        w1 = corrScore.loc[corrScore[cls[0]].idxmax()]
        w2 = corrScore.loc[corrScore[cls[1]].idxmax()]
        w3 = corrScore.loc[corrScore[cls[2]].idxmax()]

        df_1 = pd.DataFrame(w1).reset_index()
        df_2 = pd.DataFrame(w2).reset_index()
        df_3 = pd.DataFrame(w3).reset_index()

        result = pd.merge(df_1, df_2, on='index')
        result = pd.merge(result, df_3, on='index')

        result.columns = ['index', cls[0], cls[1], cls[2]]
        result = result.set_index('index')
        result = result.T
        max_result = [result.loc[result[cls[0]].idxmax()][cls[0]],
                      result.loc[result[cls[1]].idxmax()][cls[1]],
                      result.loc[result[cls[2]].idxmax()][cls[2]]]
        result["Maximum Accuracy"] = max_result
        result = result.drop([cls[0], cls[1], cls[2]], axis=1)

        result.columns.name = None

        return result

    def get_ROC_parameters(df_tmp, y):

        X_train, X_test, y_train, y_test = train_test_split(df_tmp, y, test_size=0.3, random_state=42)

        classifier_SVM_li = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                                        random_state=42))
        classifier_SVM_gu = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True,
                                                        random_state=42))
        classifier_RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)

        y_score_SVM_li = classifier_SVM_li.fit(X_train, y_train).decision_function(X_test)
        y_score_SVM_gu = classifier_SVM_gu.fit(X_train, y_train).decision_function(X_test)
        y_score_SVM_RF = classifier_RF.fit(X_train, y_train)

        fpr_li, tpr_li, _ = roc_curve(np.int64(y_test.ravel()), y_score_SVM_li.ravel())
        fpr_gu, tpr_gu, _ = roc_curve(np.int64(y_test.ravel()), y_score_SVM_gu.ravel())

        roc_auc_li = auc(fpr_li, tpr_li)
        roc_auc_gu = auc(fpr_gu, tpr_gu)

        y_pred_proba_RF = classifier_RF.predict_proba(X_test)[::, 1]
        fpr_RF, tpr_RF, _ = metrics.roc_curve(np.int64(y_test.values), y_pred_proba_RF)
        roc_auc_RF = metrics.roc_auc_score(np.int64(y_test.values), y_pred_proba_RF)

        fpr = [fpr_li, fpr_gu, fpr_RF]
        tpr = [tpr_li, tpr_gu, tpr_RF]
        roc_auc = [roc_auc_li, roc_auc_gu, roc_auc_RF]

        return fpr, tpr, roc_auc

    def venn_diagram_data(col_m1, col_m2, col_m3):
        col_uni = FeatureSelection.get_unique_columns(col_m1, col_m2, col_m3)
        overlap = FeatureSelection.get_overlap_features(col_m1, col_m2, col_m3)
        col_c = FeatureSelection.get_overlap_two(col_m1, col_m2, col_m3, overlap)

        venn_data = [col_uni, col_c, overlap]

        return venn_data

    def checkList(list1, list2):
        for word in list2:
            if word in list1:
                list1.remove(word)

        return list1

    def get_unique_columns(col_m1, col_m2, col_m3):
        col1_uni = FeatureSelection.checkList(list(col_m1), list(col_m2 + col_m3))
        col2_uni = FeatureSelection.checkList(list(col_m2), list(col_m1 + col_m3))
        col3_uni = FeatureSelection.checkList(list(col_m3), list(col_m2 + col_m1))

        col_uni = [col1_uni, col2_uni, col3_uni]

        return col_uni

    def get_overlap_features(col1, col2, col3):
        t = list(set(col1) & set(col2) & set(col3))
        return t

    def get_overlap_two(col_m1, col_m2, col_m3, overlap):
        col_m12 = list(set(col_m1) & set(col_m2))
        col_m12_uni = FeatureSelection.checkList(list(col_m12), list(overlap))

        col_m23 = list(set(col_m2) & set(col_m3))
        col_m23_uni = FeatureSelection.checkList(list(col_m23), list(overlap))

        col_m13 = list(set(col_m1) & set(col_m3))
        col_m13_uni = FeatureSelection.checkList(list(col_m13), list(overlap))

        col_c = [col_m12_uni, col_m23_uni, col_m13_uni]

        return col_c

    def get_gene_info_df(file_path):
        data = pd.read_csv(file_path, sep="\t", index_col="Symbol")
        data = data.loc[~data.index.duplicated(keep='first')]
        return data

    def get_selected_gene_info(file_path, gene_symbols):
        df = FeatureSelection.get_gene_info_df(file_path)
        df_sel = df.index.isin(gene_symbols)
        df = df[df_sel]
        return df