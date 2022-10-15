
import jsonify
import requests
import pickle
import numpy as np
import sys
import os
import re
import sklearn
import pandas as pd
from flask import Flask, render_template, url_for, flash, redirect, request, send_from_directory
from sklearn.preprocessing import StandardScaler
from os.path import join, dirname, realpath

app = Flask(__name__)

model = pickle.load(open('sepsis_rf_new.pkl', 'rb'))
UPLOAD_FOLDER = 'static/files'
UPLOAD_FILE = 'static/files/patientdata.csv'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

@app.route('/')
@app.route('/index',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/patientprediction',methods=['GET'])
def patientprediction():
    return render_template('patientprediction.html')

@app.route('/datatraining',methods=['GET'])
def datatraining():
    return render_template('datatraining.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        
        
        data = pd.read_csv(UPLOAD_FILE)
        patid = int(request.form['patid'])
        x = data.loc[data['PatientID'] == patid]
        x.drop('PatientID', inplace=True, axis=1)
        print(x)
        inp = np.array(x)
        print(inp)
        prediction = model.predict(inp)
        output="no"
        if prediction[0]==1:
            output=""
     
        return render_template('patientprediction.html', prediction_text="Patient " + " has " + output+" symtoms " +"of Sepsis" )
  
@app.route('/uploadfiles',methods=['POST'])
def uploadfiles():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
              # set the file path
            uploaded_file.save(file_path)
              # save the file
            return render_template('patientprediction.html', upload_text="Uploaded Successfully" )
        else:
            return render_template('patientprediction.html', upload_text="Upload failed" )
        
   

if __name__=='__main__':
	app.run(debug=True)
