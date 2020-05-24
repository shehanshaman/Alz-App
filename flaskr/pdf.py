from flask import Blueprint

from flaskr.auth import login_required, UserData

from flask import request
from flask import render_template

from .classes.featureSelectionClass import FeatureSelection

bp = Blueprint("pdf", __name__, url_prefix="/pdf")

#pdf data for create preprocess file
@bp.route("/preprocessing", methods=['POST', 'GET'])
@login_required
def preprocessing_pdf():
	id = request.args.get("id")

	preprocess = UserData.get_preprocess_from_id(id)

	col_sel_method_set = ['Average', 'Max', 'Min', 'Interquartile range']

	if( preprocess['col_sel_method'] == '' ):
		preprocess_data = { 
			"file_name": preprocess['file_name'], 
			"annotation_table": '-', 
			"prob_mthd": '-',
			"normalize": preprocess['scaling'],
			"imputation": preprocess['imputation'],
			"volcano_hash": preprocess['volcano_hash'],
			"fold": preprocess['fold'],
			"pvalue": preprocess['pvalue'],
			"univariate_length": preprocess['length'],
			"fr_univariate_hash": preprocess['fr_univariate_hash']			
		};
	else:
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

	return render_template("pdf/preprocess_pdf.html", data=preprocess_data, data_after_norm=preprocess['after_norm_set'], data_plot=preprocess_data_plot, clf_results=preprocess['classification_result_set'])

#pdf data for create preprocess file
@bp.route("/feature_selection", methods=['POST', 'GET'])
@login_required
def feature_selection_pdf():
	id = request.args.get("id")
	
	feature_details = UserData.get_result_from_id(id)

	feature_details_set = {
			"filename": feature_details['filename'],
			"fs_methods": feature_details['fs_methods'],
			"col_method1": feature_details['col_method1'],
			"col_method2": feature_details['col_method2'],
			"col_method3": feature_details['col_method3']		
		};

	return render_template("pdf/feature_pdf.html", feature_details=feature_details_set, venn_data = feature_details['venn_data_set'], fs_hash = feature_details['fs_hash'])

#pdf data for create preprocess file
@bp.route("/analysis", methods=['POST', 'GET'])
@login_required
def analysis_pdf():
	id = request.args.get("id")
	
	anlz_details = UserData.get_result_from_id(id)

	anlz_details_set = {
			"filename": anlz_details['filename'],
			"an_overlap_hash": anlz_details['an_overlap_hash'],
			"an_cls_hash": anlz_details['an_cls_hash'],
			"an_crr_hash": anlz_details['an_crr_hash'],
			"col_selected_method": anlz_details['col_selected_method'],
			"selected_method": anlz_details['selected_method'],
			"an_crr_1_hash": anlz_details['an_crr_1_hash'],
			"an_crr_2_hash": anlz_details['an_crr_2_hash'],
			"selected_roc_pic_hash": anlz_details['selected_roc_pic_hash'],
			"all_roc_pic_hash": anlz_details['an_crr_2_hash'],
			"col_overlapped": anlz_details['col_overlapped']
		};

	return render_template("pdf/analysis_pdf.html", anlz_details_set=anlz_details_set, corr_classification_accuracy=anlz_details['corr_classification_accuracy'], result_data_1=anlz_details['result_data_1'], result_data_2=anlz_details['result_data_2'])