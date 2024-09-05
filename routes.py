
from flask import Flask
from controller import execution

def create_app():
    app = Flask(__name__)
    app.route('/product/cv', methods=['POST', 'GET'])(execution)

    app.route('/')(lambda: "Ok,I am healthyyy!")
    return app