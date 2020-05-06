from flask import Blueprint, request
from flask import render_template
from flask import redirect
from flaskr.classes.featureSelectionClass import FeatureSelection
from flaskr.classes.preProcessClass import PreProcess
from .auth import UserData, login_required

from pathlib import Path
from werkzeug.exceptions import abort

import json

ROOT_PATH = Path.cwd()
GENE_CARD = ROOT_PATH / "flaskr" / "upload" / "Validation" / "GeneCards-SearchResults.pkl"
VALIDATION_PATH = ROOT_PATH / "flaskr" / "upload" / "Validation"
GENE_INFO_PATH = ROOT_PATH / "flaskr" / "upload" / "gene_info"

bp = Blueprint("validation", __name__, url_prefix="/val")

@bp.route("/", methods = ['GET'])
@login_required
def index():
    validation_file_name = request.args.get("file")
    result_id = request.args.get("id")

    if result_id is None:
        return redirect('../fs/val/config')

    r = UserData.get_result_from_id(result_id)
    if r is None:
        return abort(403)

    col_overlapped = r['col_overlapped'].split(',')
    col_selected_method = r['col_selected_method'].split(',')
    filename = r['filename']

    if col_overlapped is None and col_selected_method is None:
        return redirect('/an')

    col_m1 = r['col_method1'].split(',')
    col_m2 = r['col_method2'].split(',')
    col_m3 = r['col_method3'].split(',')
    method_names = r['fs_methods'].split(',')

    col_mo = list(dict.fromkeys(col_overlapped + col_selected_method))

    disease_file_path = VALIDATION_PATH / validation_file_name

    # gene_card_df = PreProcess.getDF(disease_file_path) get_gene_card_df
    gene_card_df = PreProcess.get_gene_card_df(disease_file_path)

    col_gene_card = gene_card_df.columns.tolist()

    col_m1_gene_card = get_overlap_features(col_gene_card, col_m1)
    col_m2_gene_card = get_overlap_features(col_gene_card, col_m2)
    col_m3_gene_card = get_overlap_features(col_gene_card, col_m3)

    data_available = 1
    if not col_m1_gene_card and not col_m2_gene_card and not col_m3_gene_card:
        data_available = 0

    col_dist_gene_card = get_overlap_features(col_gene_card, col_mo)

    dis_gene_card = gene_card_df[col_dist_gene_card]

    col_gene_card = [col_m1_gene_card, col_m2_gene_card, col_m3_gene_card, col_mo, col_dist_gene_card]

    venn_data = FeatureSelection.venn_diagram_data(col_m1_gene_card, col_m2_gene_card, col_m3_gene_card)

    #Get gene info
    gene_info_path = GENE_INFO_PATH / "Homo_sapiens.gene_info"
    unique_genes = list(set(col_m1 + col_m2 + col_m3))

    gene_info_df = FeatureSelection.get_selected_gene_info(gene_info_path, unique_genes)
    gene_info = gene_info_df.to_json(orient='index')

    gene_info = json.loads(gene_info)

    gene_name_list = list(gene_info_df.index)

    dis_gene_card = dis_gene_card.T
    dis_gene_card.columns.name = dis_gene_card.index.name
    dis_gene_card.index.name = None
    dis_gene_card = dis_gene_card.sort_values(by='Relevance score', ascending=False)

    return render_template("validation/index.html", col_gene_card = col_gene_card, method_names = method_names,
                           tables=[dis_gene_card.to_html(classes='data')], venn_data=venn_data, filename=filename,
                           result_id = result_id, gene_info = gene_info, gene_name_list = gene_name_list, data_available = data_available)

def get_overlap_features(col1, col2):
    t = list(set(col1) & set(col2))
    return t