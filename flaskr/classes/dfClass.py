class DF(object):
    def __init__(self, file_name: str, path: str, anno_tbl: str, col_sel_method: str, merge_df: str, symbol_df: str, avg_symbol_df:str, reduce_df:str):
        self.file_name = file_name
        self.path = path
        self.anno_tbl = anno_tbl
        self.col_sel_method = col_sel_method
        self.merge_df = merge_df
        self.symbol_df = symbol_df
        self.avg_symbol_df = avg_symbol_df
        self.reduce_df = reduce_df


    def setMergeDF(self, df):
        self.merge_df = df # Probes and Dataset merge not pre-process

    def setSymbolDF(self, df):
        self.symbol_df = df #according to symbol | Normalize | Remove null rows

    def setAvgSymbolDF(self, df):
        self.avg_symbol_df = df

    def setReduceDF(self, df):
        self.reduce_df = df

# def getAnnoTblName(self):
# 	return self.anno_tbl

# def getPath(self):
# 	return self.path

# def getColSelectionMethod(self):
# 	return self.col_sel_method