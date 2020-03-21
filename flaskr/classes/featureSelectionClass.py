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