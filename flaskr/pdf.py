from flask import Blueprint

from flaskr.auth import login_required, UserData

from flask import request
from flask import render_template

bp = Blueprint("pdf", __name__, url_prefix="/pdf")

#pdf data for create preprocess file
@bp.route("/preprocessing", methods=['POST', 'GET'])
@login_required
def preprocessing_pdf():
	id = request.args.get("id")

	preprocess = UserData.get_preprocess_from_id(id)

	# preprocess_data = { 
	# 		"file_name": preprocess['file_name'], 
	# 		"annotation_table": (preprocess['annotation_table']).replace('/AnnotationTbls/',''), 
	# 		"prob_mthd": preprocess['col_sel_method'],
	# 		"normalize": preprocess['scaling'],
	# 		"imputation": preprocess['imputation'],
	# 		"after_norm_details": preprocess['after_norm_set'],
	# 		"volcano_hash": preprocess['volcano_hash'],
	# 		"fold": preprocess['fold'],
	# 		"pvalue": preprocess['pvalue'],
	# 		"univariate_length": preprocess['length'],
	# 		"fr_univariate_hash": preprocess['fr_univariate_hash'],
	# 		"classification_result_set": preprocess['classification_result_set']			
	# 	};

	col_sel_method_set = ['Average', 'Max', 'Min', 'Interquartile range']

	preprocess_data = { 
		"file_name": preprocess['file_name'], 
		"annotation_table": (preprocess['annotation_table']).replace('/AnnotationTbls/',''), 
		"prob_mthd": col_sel_method_set[int(preprocess['col_sel_method'])-1],
		"normalize": preprocess['scaling'],
		"imputation": preprocess['imputation'],
		"volcano_hash": preprocess['volcano_hash'],
		"fold": preprocess['fold'],
		"pvalue": preprocess['pvalue'],
		"univariate_length": preprocess['length'],
		"fr_univariate_hash": preprocess['fr_univariate_hash']			
	};

	preprocess_data_plot = {
			"volcano_hash": preprocess['volcano_hash'],
			"fold": preprocess['fold'],
			"pvalue": preprocess['pvalue'],
			"univariate_length": preprocess['length'],
			"fr_univariate_hash": preprocess['fr_univariate_hash']		
		};

	return render_template("pdf/index.html", data=preprocess_data, data_after_norm=preprocess['after_norm_set'], data_plot=preprocess_data_plot, clf_results=preprocess['classification_result_set'])