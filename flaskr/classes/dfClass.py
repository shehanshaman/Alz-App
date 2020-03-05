class DF:
    def __init__(self, path, anno_tbl, col_sel_method):
        self.path = path
        self.anno_tbl = anno_tbl
        self.col_sel_method = col_sel_method

    def setDF(self, df):
        self.df = df

# def getAnnoTblName(self):
# 	return self.anno_tbl

# def getPath(self):
# 	return self.path

# def getColSelectionMethod(self):
# 	return self.col_sel_method