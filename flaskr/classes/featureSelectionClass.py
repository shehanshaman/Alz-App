from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

class FeatureSelection:

    def PCA(df_200F, len):
        y = df_200F["class"]
        X = df_200F.drop("class",1).values
        initial_feature_names = df_200F.columns.values

        # 160 samples with 2312 features
        train_features = X

        model = PCA(n_components=55).fit(train_features)  # n_components=63
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