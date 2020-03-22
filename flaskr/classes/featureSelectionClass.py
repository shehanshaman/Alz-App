from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

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

    def getTop3ClassificationResults(X_train, X_test, y_train, y_test):
        svm_li_clf = svm.SVC(kernel='linear')  # Linear Kernel
        score_svm_li = cross_val_score(svm_li_clf, X_train, y_train, cv=3)
        svm_li_clf.fit(X_train, y_train)

        svm_rbf_clf = SVC(kernel="rbf", gamma="auto", C=1)
        score_svm_rbf = cross_val_score(svm_rbf_clf, X_train, y_train, cv=3)
        svm_rbf_clf.fit(X_train, y_train)

        RF_clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
        score_RF = cross_val_score(RF_clf, X_train, y_train, cv=3)
        RF_clf.fit(X_train, y_train)

        y_pred_svm_li = svm_li_clf.predict(X_test)
        y_pred_svm_rbf = svm_rbf_clf.predict(X_test)
        y_pred_RF = RF_clf.predict(X_test)

        m = np.array([[round(metrics.accuracy_score(np.int64(y_test.values), y_pred_svm_rbf) * 100, 2),
                       round(score_svm_rbf.mean() * 100, 2)],
                      [round(metrics.accuracy_score(np.int64(y_test.values), y_pred_svm_li) * 100, 2),
                       round(score_svm_li.mean() * 100, 2)],
                      [round(metrics.accuracy_score(np.int64(y_test.values), y_pred_RF) * 100, 2),
                       round(score_svm_li.mean() * 100, 2)]])

        return m

    def getTop3ClassificationResults_by_df(df, y):
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)
        m = FeatureSelection.getTop3ClassificationResults(X_train, X_test, y_train, y_test)
        classifier_result_df = pd.DataFrame(m, columns=["Testing", "Training"])
        classifiers = ["SVM + Gaussian kernel", "SVM + Linear kernel", "Random Forest"]
        classifier_result_df["index"] = classifiers
        classifier_result_df = classifier_result_df.set_index('index')
        classifier_result_df.index.name = None

        return classifier_result_df

    def getSummaryFeatureSelection(df_50F_PCA, df_50F_FI, df_50F_RF, y, rows):
        X_train, X_test, y_train, y_test = train_test_split(df_50F_PCA, y, test_size=0.3, random_state=42)
        m_pca = FeatureSelection.getTop3ClassificationResults(X_train, X_test, y_train, y_test)

        X_train, X_test, y_train, y_test = train_test_split(df_50F_FI, y, test_size=0.3, random_state=42)
        m_fi = FeatureSelection.getTop3ClassificationResults(X_train, X_test, y_train, y_test)

        X_train, X_test, y_train, y_test = train_test_split(df_50F_RF, y, test_size=0.3, random_state=42)
        m_rf = FeatureSelection.getTop3ClassificationResults(X_train, X_test, y_train, y_test)

        # rows = ["classifier", "PCA", "Random Forest", "Extra Tree"]
        rows.insert(0, "classifier")
        cla = ["SVM + Gaussian kernel", "SVM + linear kerne", "Random forest"]

        data_testing = np.array([cla, m_pca[:, 0], m_fi[:, 0], m_rf[:, 0]])
        results_testing = pd.DataFrame(data=data_testing, index=rows).transpose()

        data_training = np.array([cla, m_pca[:, 1], m_fi[:, 1], m_rf[:, 1]])
        results_training = pd.DataFrame(data=data_training, index=rows).transpose()

        results_testing = results_testing.set_index('classifier')
        results_training = results_training.set_index('classifier')

        results_testing = results_testing.astype(float)
        results_training = results_training.astype(float)

        return results_testing, results_training

    def getFeatureSummary(df, y, col_uni, overlap, rows):
        list0 = ["SVM + Gaussian kernel", "SVM + Linear kernel", "Random forest"]

        count = [len(col_uni[0]), len(col_uni[1]), len(col_uni[2]), len(overlap)]

        diff_df = df[col_uni[0]]

        X_train, X_test, y_train, y_test = train_test_split(diff_df, y, test_size=0.3, random_state=42)
        list1 = FeatureSelection.getTop3ClassificationResults(X_train, X_test, y_train, y_test)[:, 0]  # same dataset

        diff_df = df[col_uni[1]]

        X_train, X_test, y_train, y_test = train_test_split(diff_df, y, test_size=0.3, random_state=42)
        list2 = FeatureSelection.getTop3ClassificationResults(X_train, X_test, y_train, y_test)[:, 0]  # same dataset

        diff_df = df[col_uni[2]]

        X_train, X_test, y_train, y_test = train_test_split(diff_df, y, test_size=0.3, random_state=42)
        list3 = FeatureSelection.getTop3ClassificationResults(X_train, X_test, y_train, y_test)[:, 0]  # same dataset

        diff_df = df[overlap]

        X_train, X_test, y_train, y_test = train_test_split(diff_df, y, test_size=0.3, random_state=42)
        list4 = FeatureSelection.getTop3ClassificationResults(X_train, X_test, y_train, y_test)[:, 0]  # same dataset

        rows.insert(0, "classifier")
        rows.insert(4, "Overlap")
        rows.pop()
        # rows = ["classifier", "PCA", "Random Forest", "Extra Tree", "Overlap"]

        data = np.array([list0, list1, list2, list3, list4])
        results = pd.DataFrame(data=data, index=rows).transpose()

        results = results.set_index('classifier')

        df_count = pd.DataFrame({'id': rows[1:5], 'val': count})

        return results, df_count



    def returnScoreDataFrameModels(dataFrame, y, len):
        lists1 = []
        lists2 = []
        lists3 = []

        for i in FeatureSelection.get_range_array(len):

            lists1.append(FeatureSelection.svmLinear(dataFrame.iloc[:, 0:(i)], y))
            lists2.append(FeatureSelection.svmGaussian(dataFrame.iloc[:, 0:(i)], y))
            lists3.append(FeatureSelection.randomForest(dataFrame.iloc[:, 0:(i)], y))

        rows = ["svmLinear", "svmGaussian", "randomForest"]

        data = np.array([lists1, lists2, lists3])
        modelScore = pd.DataFrame(data=data, index=rows).transpose()

        return modelScore

    def get_range_array(len):
        array = [100, 90, 80, 70, 60, 50, 40, 30, 20, 15, 10, 5, 3, 2, 1]
        return array[array.index(FeatureSelection.roundupten(len)):]

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

    def returnScoreDataFrame(dataFrame, y):
        lists0 = []
        lists1 = []
        lists2 = []
        lists3 = []
        lists4 = []

        i = 1

        while i >= 0.6:
            # for i in range(0,50):
            df_tmp = dataFrame
            df_tmp = FeatureSelection.getSelectedDF(df_tmp, df_tmp.corr(), i)

            lists0.append(i)
            lists1.append(FeatureSelection.svmLinear(df_tmp, y))
            lists2.append(FeatureSelection.svmGaussian(df_tmp, y))
            lists3.append(FeatureSelection.randomForest(df_tmp, y))
            lists4.append(len(df_tmp.columns))

            i = i - 0.0025

        rows = ["i", "svmLinear", "svmGaussian", "randomForest", "No of features"]

        data = np.array([lists0, lists1, lists2, lists3, lists4])
        results = pd.DataFrame(data=data, index=rows).transpose()

        return results

    def get_max_corr_scores(corrScore):
        w1 = corrScore.loc[corrScore['svmLinear'].idxmax()]
        w2 = corrScore.loc[corrScore['svmGaussian'].idxmax()]
        w3 = corrScore.loc[corrScore['randomForest'].idxmax()]

        df_1 = pd.DataFrame(w1).reset_index()
        df_2 = pd.DataFrame(w2).reset_index()
        df_3 = pd.DataFrame(w3).reset_index()

        result = pd.merge(df_1, df_2, on='index')
        result = pd.merge(result, df_3, on='index')

        result.columns = ['index', 'svmLinear', 'svmGaussian', 'randomForest']
        result = result.set_index('index')
        result = result.T
        max_result = [result.loc[result['svmLinear'].idxmax()]['svmLinear'],
                      result.loc[result['svmGaussian'].idxmax()]['svmGaussian'],
                      result.loc[result['randomForest'].idxmax()]['randomForest']]
        result["Maximum Accuracy"] = max_result
        result = result.drop(["svmLinear", "svmGaussian", "randomForest"], axis=1)

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

        # plt.plot(fpr_li, tpr_li, label="SVM linear, auc=" + str(round(roc_auc_li, 2)))
        # plt.plot(fpr_gu, tpr_gu, label="SVM gaussian, auc=" + str(round(roc_auc_gu, 2)))
        # plt.plot(fpr_RF, tpr_RF, label="Random forest, auc=" + str(round(roc_auc_RF, 2)))
        #
        # plt.ylabel("Sensitivity")
        # plt.xlabel("1 - Specificity")
        #
        # plt.legend(loc="lower right")

        return fpr, tpr, roc_auc