import pandas as pd
import numpy as np

import pandas as pd
from flask import jsonify
from sklearn import preprocessing

from scipy import stats
import statistics
from scipy.stats import ttest_ind

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
		min_max_scaler = preprocessing.MinMaxScaler()
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

	def getDfDetails(df):
		# df.set_index('ID', inplace=True)
		# df = df.T
		# print(df.head())
		# x = df.drop("class",1)
		x =df
		shape = x.shape
		min = x.min().min()
		max = x.max().max()
		unique_probes = x['ID'].unique().shape
		null_count = x.isnull().sum()

		ds = {'shape': shape, 'min': min, 'max': max, 'unique_probes': unique_probes, 'null_count':null_count}
		return jsonify(ds)

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

	# def get_reduced_df_from_pvalues(df_path, class_path, pvalue, foldChange):
	# 	df = PreProcess.set_class_to_df(df_path, class_path)
	# 	df_normal, df_AD = PreProcess.split_df_by_class(df)
	# 	filtered_index = PreProcess.sort_pValues(df_AD, df_normal, pvalue, foldChange)
	# 	sorted_dataframe = df.filter(df.columns[filtered_index])
	# 	return sorted_dataframe
    #
	# def get_reduced_feature_count_from_pvalues(df_path, class_path, pvalue, foldChange):
	# 	df = PreProcess.set_class_to_df(df_path, class_path)
	# 	df_normal, df_AD = PreProcess.split_df_by_class(df)
	# 	filtered_index = PreProcess.sort_pValues(df_AD, df_normal, pvalue, foldChange)
	# 	return len(filtered_index)
