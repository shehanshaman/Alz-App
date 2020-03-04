import pandas as pd
import numpy as np

class PreProcess:
	
	def getDF(name):
		df = pd.read_pickle(name)
		df = df.iloc[[0, 2], [1, 3]]
		return df

	def getProbeDF(name):
		probes = pd.read_csv(name)
		probes = probes[['Gene Symbol', 'ID']]
		return probes.head()

