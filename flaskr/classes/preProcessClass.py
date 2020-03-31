import pandas as pd
import numpy as np

import pandas as pd
from flask import jsonify
from sklearn import preprocessing

from scipy import stats
import statistics
from scipy.stats import ttest_ind

import json

class PreProcess:
	
	def getDF(name):
		df = pd.read_pickle(name)
		# df = df.iloc[[0, 2], [1, 3]]
		return df

	def saveDF(df, path):
		df.to_pickle(path)

	def getProbeDF(name):
		probes = pd.read_csv(name)
		probes = probes[['Gene Symbol', 'ID']]
		return probes

	def mergeDF(df_path, probe_path):
		df = PreProcess.getDF(df_path)
		df_T = df.T.reset_index()
		df_T = df_T.rename(columns={'index': 'ID'})
		probes = PreProcess.getProbeDF(probe_path)
		df_merge = pd.merge(df_T, probes, on='ID')

		return df_merge

	def rmNullRows(df_merge):
		df_merge_rm_null = df_merge.dropna(how='any',axis=0)
		return df_merge_rm_null

	def df2float(df_merge_rm_null):
		df_merge_rm_null_float = df_merge_rm_null.iloc[:,1:df_merge_rm_null.columns.shape[0]-1].astype(float)
		return df_merge_rm_null_float

	def dfNormSKlearn(df_merge_rm_null_float, df_merge_rm_null):
		x = df_merge_rm_null_float.values #returns a numpy array
		min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 15))
		x_scaled = min_max_scaler.fit_transform(x)
		df_norm = pd.DataFrame(x_scaled)

		df_symbol = pd.DataFrame()
		df_val = pd.DataFrame()

		df_symbol['Gene Symbol'] = df_merge_rm_null['Gene Symbol']

		df_val[df_merge_rm_null_float.columns]  = df_norm

		df_symbol.reset_index(drop=True, inplace=True)
		df_val.reset_index(drop=True, inplace=True)
		df_symbol = pd.concat([df_symbol, df_val], axis=1)
		
		return df_symbol

	def step3(df_merge, norm_method, imputation_method):
		df_merge_rm_null = df_merge

		if imputation_method == 'drop':
			df_merge_rm_null = PreProcess.rmNullRows(df_merge)

		df_merge_rm_null_float = PreProcess.df2float(df_merge_rm_null)

		if norm_method == 'sklearn':
			df_symbol = PreProcess.dfNormSKlearn(df_merge_rm_null_float, df_merge_rm_null)

		return df_symbol

	def probe2Symbol(df_symbol):
		df_avg_symbol = df_symbol.groupby(['Gene Symbol']).agg([np.average])
		df_avg_symbol.reset_index(drop=False, inplace=True)
		# df_avg_symbol = df_avg_symbol.drop(['index'], axis = 1)
		df_avg_symbol.columns = df_symbol.columns

		return df_avg_symbol

	def split_df_by_class(df):
		df['class'] = df['class'].astype(str).astype(int)
		df_normal = df[df['class'] == 0]
		df_AD = df[df['class'] == 1]
		return df_normal, df_AD

	def set_class_to_df(df_path, class_path):
		df = PreProcess.getDF(df_path)
		df = df.set_index(["Gene Symbol"])
		df = df.T

		df_class = PreProcess.getDF(class_path)
		df['class'] = df_class['class']

		return df

	def get_pvalue_fold_df(df_path, class_path):
		df = PreProcess.set_class_to_df(df_path, class_path)
		df_normal, df_AD = PreProcess.split_df_by_class(df)

		pValues = []
		fold_change = []

		df_AD_trans = df_AD.transpose()
		df_normal_trans = df_normal.transpose()
		listAD = df_AD_trans.values.tolist()
		listNormal = df_normal_trans.values.tolist()

		for i in range(len(listAD) - 1):  # For each gene :
			ttest, pval = ttest_ind(listAD[i], listNormal[i])  # calculating p values for each gene using ttest
			mean_AD = statistics.mean(listAD[i])
			mean_Normal = statistics.mean(listNormal[i])
			fold = (mean_AD - mean_Normal)
			fold_change.append(fold)
			pValues.append(pval)

		p_fold_df = pd.DataFrame({'fold': fold_change, 'pValues': pValues}, columns=["fold", "pValues"])

		return p_fold_df

	def get_filtered_df_pvalue(p_fold_df, df_path, pvalue, foldChange):
		df = PreProcess.getDF(df_path)
		df = df.set_index(["Gene Symbol"])
		df = df.T

		p_fold_df['is_selected'] = (abs(p_fold_df['fold']) > foldChange) & (p_fold_df['pValues'] < pvalue)
		sorted_dataframe = df.filter(df.columns[p_fold_df['is_selected']])

		return sorted_dataframe

	def get_filtered_df_count_pvalue(p_fold_df, pvalue, foldChange):
		p_fold_df['is_selected'] = (abs(p_fold_df['fold']) > foldChange) & (p_fold_df['pValues'] < pvalue)

		return p_fold_df['is_selected'].sum()


	#DF details to json
	def get_df_details(df, y):
		d = df.drop(["ID", "Gene Symbol"], axis=1)

		x = '{ "Number of features":' + str(d.shape[0]) + ', "Number of samples": ' + str(d.shape[1]) + ', "min":' + str(
			round(d.min().min())) + ', "max":' + str(round(d.max().max())) + '}'
		z = json.loads(x)

		x = {"Number of unique Gene Symbols": str(df['Gene Symbol'].nunique()),
			 "Number of null values in Gene Symbols": str(df['Gene Symbol'].isnull().sum()),
			 "Number of null values in data": str(d.isnull().sum().sum())}
		z.update(x)

		s = y.value_counts()
		x = {"Positive": str(s[1]), "Negative": str(s[0])}
		z.update(x)

		return z

	def add_details_json(z, df, r):
		d = df.drop(["Gene Symbol"], axis=1)
		x = '{ "Number of features":' + str(d.shape[0]) + ', "Number of samples": ' + str(d.shape[1]) + ', "min":' + str(round(d.min().min(), 3)) + ', "max":' + str(round(d.max().max(),3)) + '}'
		new_z = json.loads(x)
		w = {r: new_z}
		z.update(w)

		return z

