import pandas as pd

df = pd.read_csv('GSE5281-GPL570.zip')
df = df.set_index(["ID"])
df.index.name = None
df.columns.name = "ID"
df.to_pickle('GSE5281-GPL570.pkl')
