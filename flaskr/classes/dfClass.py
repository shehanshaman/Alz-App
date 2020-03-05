class DF:
    def __init__(self, path, anno_tbl, col_sel_method):
        self.path = path
        self.anno_tbl = anno_tbl
        self.col_sel_method = col_sel_method

    def setMergeDF(self, df):
        self.merge_df = df # Probes and Dataset merge not pre-process

    def setSymbolDF(self, df):
        self.symbol_df = df #according to symbol | Normalize | Remove null rows

    def setAvgSymbolDF(self, df):
        self.avg_symbol_df = df

# def getAnnoTblName(self):
# 	return self.anno_tbl

# def getPath(self):
# 	return self.path

# def getColSelectionMethod(self):
# 	return self.col_sel_method