# Auto-generated file.
from flask import Flask
from TimeToDoor.logger import logging

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    logging.info("Home route accessed")
    return "Welcome to TimeToDoor API!"

if __name__ == "__main__":
    logging.info("Starting TimeToDoor API")
    app.run(debug=True)
