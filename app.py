from Item.routes import create_app
from bs4 import BeautifulSoup
from flask_cors import CORS

app = create_app()
CORS(app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)