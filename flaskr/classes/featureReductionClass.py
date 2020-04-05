import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from matplotlib.figure import Figure

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class FeatureReduction:

    def getScoresFromUS(df):
        X = df.drop("class", 1)  # Feature Matrix
        y = df["class"]  # Target Variable

        bestfeatures = SelectKBest(score_func=chi2, k=10)
        fit = bestfeatures.fit(X, y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)

        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']

        featureScores = featureScores.nlargest(df.shape[1]-1, 'Score')

        return featureScores

    def getSelectedFeatures(df, len, y):
        df["class"] = y  # Target Variable
        featureScores = FeatureReduction.getScoresFromUS(df)
        selectedFeatures = featureScores.nlargest(len, 'Score')
        pd.options.mode.chained_assignment = None
        df_200F = df[selectedFeatures['Specs'].values]
        df_200F['class'] = y

        return df_200F

    def create_figure(selectedFeatures, length):
        fig = Figure(figsize=(12, 8))
        axis = fig.add_subplot(1, 1, 1)
        axis.set_ylabel('Score')
        axis.set_xlabel('Specs')
        axis.set_title('Univariate Selection: ' + str(length) + " features")

        axis.plot( selectedFeatures["Specs"], selectedFeatures["Score"], label='linear')
        return fig

    def get_classification_results(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        gnb_clf = GaussianNB()
        score_gnb = cross_val_score(gnb_clf, X_train, y_train, cv=3)
        gnb_clf.fit(X_train, y_train)

        dt_clf = DecisionTreeClassifier()
        score_dt = cross_val_score(dt_clf, X_train, y_train, cv=3)
        dt_clf.fit(X_train, y_train)

        knn_clf = KNeighborsClassifier(n_neighbors=2)
        score_knn = cross_val_score(knn_clf, X_train, y_train, cv=3)
        knn_clf.fit(X_train, y_train)

        svm_li_clf = svm.SVC(kernel='linear')  # Linear Kernel
        score_svm_li = cross_val_score(svm_li_clf, X_train, y_train, cv=3)
        svm_li_clf.fit(X_train, y_train)

        svm_rbf_clf = SVC(kernel="rbf", gamma="auto", C=1)
        score_svm_rbf = cross_val_score(svm_rbf_clf, X_train, y_train, cv=3)
        svm_rbf_clf.fit(X_train, y_train)

        RF_clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
        score_RF = cross_val_score(RF_clf, X_train, y_train, cv=3)
        RF_clf.fit(X_train, y_train)

        # making predictions on the testing set
        y_pred_gnb = gnb_clf.predict(X_test)
        y_pred_dt = dt_clf.predict(X_test)
        y_pred_knn = knn_clf.predict(X_test)
        y_pred_svm_li = svm_li_clf.predict(X_test)
        y_pred_svm_rbf = svm_rbf_clf.predict(X_test)
        y_pred_RF = RF_clf.predict(X_test)

        testing = [round(metrics.accuracy_score(np.int64(y_test.values), y_pred_gnb) * 100, 2),
                   round(metrics.accuracy_score(np.int64(y_test.values), y_pred_dt) * 100, 2),
                   round(metrics.accuracy_score(np.int64(y_test.values), y_pred_knn) * 100, 2),
                   round(metrics.accuracy_score(np.int64(y_test.values), y_pred_svm_rbf) * 100, 2),
                   round(metrics.accuracy_score(np.int64(y_test.values), y_pred_svm_li) * 100, 2),
                   round(metrics.accuracy_score(np.int64(y_test.values), y_pred_RF) * 100, 2)]

        training = [round(score_gnb.mean() * 100, 2), round(score_dt.mean() * 100, 2), round(score_knn.mean() * 100, 2),
                    round(score_svm_rbf.mean() * 100, 2), round(score_svm_li.mean() * 100, 2), round(score_RF.mean() * 100, 2)]

        classifiers = ["Gaussian Naive Bayes", "Decision Tree", "Nearest Neighbors", "SVM + Gaussian kernel", "SVM + linear kernel",
                       "Random Forest"]

        df = pd.DataFrame({"Classifiers": classifiers, "Testing": testing, "Training": training}, columns=["Classifiers", "Testing", "Training"])

        return df