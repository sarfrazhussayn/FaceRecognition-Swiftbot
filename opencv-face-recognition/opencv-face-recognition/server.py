from flask import Flask, render_template
import subprocess
from subprocess import call

app = Flask(__name__)

#Root for main (basically no path after the host)
@app.route("/")
#HomePage Rendered
def main():
	return render_template('homepage.html')

@app.route("/Train")
def train():
	return render_template('test.html')

@app.route("/run_program")
def run_program():
	#exec(compile("recognize_video.py",) "--detector .\face_detection_model\ --embedding-model .\openface_nn4.small2.v1.t7 --recognizer .\output\recognizer.pickle --le .\output\le.pickle"))
	#exec("recognize_video.py --detector .\face_detection_model\ --embedding-model .\openface_nn4.small2.v1.t7 --recognizer .\output\recognizer.pickle --le .\output\le.pickle")
	return 'Good'


if __name__ == "__main__":
	app.run(debug=True,host="127.0.0.1",port=80)