import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

class FeatureReduction:

    def univeriateSelection(self, df):
        X = df.drop("class", 1)  # Feature Matrix
        y = df["class"]  # Target Variable

        bestfeatures = SelectKBest(score_func=chi2, k=10)
        fit = bestfeatures.fit(X, y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)

        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']
        selectedFeatures = featureScores.nlargest(200, 'Score')

        pd.options.mode.chained_assignment = None

        df_200F = df[selectedFeatures['Specs'].values]
        df_200F['class'] = y
        # df_200F.head()

        return df_200F