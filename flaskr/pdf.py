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

	preprocess_data = { 
			"file_name": preprocess['file_name'], 
			"annotation_table": (preprocess['annotation_table']).replace('/AnnotationTbls/',''), 
			"prob_mthd": preprocess['col_sel_method'],
			"normalize": preprocess['scaling'],
			"imputation": preprocess['imputation']
		};

	return render_template("pdf/index.html", data=preprocess_data)