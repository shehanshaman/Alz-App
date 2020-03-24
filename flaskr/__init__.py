import os

from flask import Flask, render_template, session, request

from .classes.app_alz import alz
from flaskr.classes.preProcessClass import PreProcess

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
USER_PATH = ROOT_PATH + "\\upload\\users\\"

def create_app(test_config=None):
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        # a default secret that should be overridden by instance config
        SECRET_KEY="dev",
        # store the database in the instance folder
        DATABASE=os.path.join(app.instance_path, "flaskr.sqlite"),

        APP_ALZ = alz(),

    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.update(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route("/view/", methods = ["POST"])
    def view_df():
        user_id = session.get("user_id")
        selected_file = request.form["selected_file"]
        df = PreProcess.getDF(USER_PATH + str(user_id) + "\\" + selected_file)
        return render_template('preprocess/tableVIew.html', tables=[df.to_html(classes='data')], titles=df.columns.values)

    # register the database commands
    from flaskr import db

    db.init_app(app)

    # apply the blueprints to the app
    from flaskr import auth, blog, preprocess, visualization, fs, analze, validation, modeling

    app.register_blueprint(auth.bp)
    app.register_blueprint(blog.bp)
    app.register_blueprint(preprocess.bp)
    app.register_blueprint(visualization.bp)
    app.register_blueprint(fs.bp)
    app.register_blueprint(analze.bp)
    app.register_blueprint(validation.bp)
    app.register_blueprint(modeling.bp)

    # make url_for('index') == url_for('blog.index')
    # in another app, you might define a separate main index here with
    # app.route, while giving the blog blueprint a url_prefix, but for
    # the tutorial the blog will be the main index
    app.add_url_rule("/", endpoint="index")

    return app
