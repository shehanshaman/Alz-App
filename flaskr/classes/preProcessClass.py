import pandas as pd
import numpy as np

import pandas as pd
from flask import jsonify
from sklearn import preprocessing

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

	def step3(df_merge):
		df_merge_rm_null = PreProcess.rmNullRows(df_merge)
		df_merge_rm_null_float = PreProcess.df2float(df_merge_rm_null)
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