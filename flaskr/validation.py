import os

import pandas as pd
from flask import Blueprint, session, g, request
from flask import render_template
from flask import redirect
from flaskr.classes.featureSelectionClass import FeatureSelection

from flaskr.classes.preProcessClass import PreProcess
from flaskr.classes.validation import ValidateUser
from .auth import UserData, login_required

from pathlib import Path

ROOT_PATH = Path.cwd()
GENE_CARD = ROOT_PATH / "flaskr" / "upload" / "Validation" / "GeneCards-SearchResults.pkl"

bp = Blueprint("validation", __name__, url_prefix="/val")

@bp.route("/", methods = ['GET'])
@login_required
def index():
    result_id = request.args.get("id")

    if result_id is None:
        return redirect('../fs/val/config')

    r = UserData.get_result_from_id(result_id)

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

    gene_card_df = PreProcess.getDF(GENE_CARD)
    col_gene_card = gene_card_df.columns.tolist()

    col_m1_gene_card = get_overlap_features(col_gene_card, col_m1)
    col_m2_gene_card = get_overlap_features(col_gene_card, col_m2)
    col_m3_gene_card = get_overlap_features(col_gene_card, col_m3)

    col_dist_gene_card = get_overlap_features(col_gene_card, col_mo)

    dis_gene_card = gene_card_df[col_dist_gene_card]

    col_gene_card = [col_m1_gene_card, col_m2_gene_card, col_m3_gene_card, col_mo, col_dist_gene_card]

    venn_data = FeatureSelection.venn_diagram_data(col_m1_gene_card, col_m2_gene_card, col_m3_gene_card)

    return render_template("validation/index.html", col_gene_card = col_gene_card, method_names = method_names,
                           tables=[dis_gene_card.head().to_html(classes='data')], venn_data=venn_data, filename=filename,
                           result_id = result_id)

def get_overlap_features(col1, col2):
    t = list(set(col1) & set(col2))
    return t