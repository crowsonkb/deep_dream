import flask
app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if flask.request.method == 'POST':
        return 'Do something here.'
    return flask.render_template('index.html')
