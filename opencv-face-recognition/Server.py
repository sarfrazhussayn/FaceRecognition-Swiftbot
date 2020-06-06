import os
import urllib.request
from app import app
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
# from werkzeug.utils import secure_filename
from Recognizer import indexe

ALLOWED_EXTENSIONS = set(['mp4','txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
	# check if the post request has the file part
	if 'file' not in request.files:
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 405
		return resp
	file = request.files['file']
	if file.filename == '':
		resp = jsonify({'message' : 'No file selected for uploading'})
		resp.status_code = 406
		return resp
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename) #Creates Files name
		basedir = os.path.abspath(os.path.dirname(__file__)) #Fetches Base DIR
		file.save(os.path.join(basedir,'./UPLOAD_FOLDER', filename)) #Saves File on the UPLOAD Destination
		
		########OBSOLETE#########
		#destination = file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'], filename))
		#print(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'], filename))
		#print(destination)
		#########################
		
		#Runs Recognizer on the Last Uploaded File
		#print(indexe("./UPLOAD_FOLDER/"+str(filename)))
		#Will print the Response of Facial Recognition
		
		print(filename)
		
		resp = jsonify({'message' : 'File successfully uploaded'})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify({'message' : 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
		resp.status_code = 400
		return resp

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000, debug=True)