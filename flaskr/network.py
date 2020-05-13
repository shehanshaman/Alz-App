from flask import Blueprint, g, abort, session, redirect, url_for
from flask import render_template
from flaskr.db import get_db

from flaskr.auth import login_required, UserData
from .classes.preProcessClass import PreProcess
from pathlib import Path
import numpy as np

from sklearn import cluster, covariance, manifold

bp = Blueprint("network", __name__, url_prefix="/net")

ROOT_PATH = Path.cwd()
TEST_FILE = ROOT_PATH / "flaskr" / "upload" / "users" / "1" / "GSE5281_DE_200.plk"

@bp.route("/")
# @login_required
def index():
    df = PreProcess.getDF(TEST_FILE)
    df = df.drop(['class'], axis=1)
    df = df.iloc[:, 0:10]

    names = df.columns.values
    X = df.values

    edge_model = covariance.GraphicalLassoCV()
    edge_model.fit(X)

    # _, labels = cluster.affinity_propagation(edge_model.covariance_)
    # n_labels = labels.max()

    # for i in range(n_labels + 1):
    #     print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))

    # node_position_model = manifold.LocallyLinearEmbedding(
    #     n_components=2, eigen_solver='dense', n_neighbors=6)
    #
    # embedding = node_position_model.fit_transform(X.T).T

    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

    start_idx, end_idx = np.where(non_zero)

    node = []
    i = 0
    for name in names:
        data_set = { "id": i, "label": name, "group": 1 }
        i = i + 1
        node.append(data_set)

    # print(node)
    edges = []
    for x in range(len(start_idx)):
        link = {"from": str(start_idx[x]), "to": str(end_idx[x])}
        edges.append(link)
    # print(edges)

    return render_template("network/index.html", names = names, start_idx=start_idx, end_idx=end_idx, node=node, edges=edges)