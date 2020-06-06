import os
from flask import Flask, render_template, request
from Recognizer import indexe
from live import indexer
#import cam

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        # live = indexer()
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        # x = indexe(destination)
        # print(x)
    print(request.files.getlist("file"))

    return render_template("complete.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=4555, debug=True)