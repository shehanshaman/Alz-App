import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from matplotlib.figure import Figure

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