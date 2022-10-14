
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

@app.route('/',methods=['GET'])
@app.route('/index',methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/datatraining',methods=['GET','POST'])
def datatraining():
    return render_template('datatraining.html',title='Data Training')

@app.route('/patientprediction',methods=['GET','POST'])
def patientprediction():
    return render_template('patientprediction.html',title='Patient Prediction')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        data = pd.read_csv(UPLOAD_FILE)
        patid = request.form['patid']
        x = data.loc[data['PatientID'] == patid]
        #x.drop('PatientID', inplace=True, axis=1)
        x = x.iloc[:, 1:]
        
        inp = np.array(x).reshape((1, -1))
        prediction = model.predict(inp)
        output="no"
        if prediction[0]==1:
            output=""
     
        return render_template('patientprediction.html', prediction_text="Patient " +patid+ " has " + output+" symtoms " +x+"of Sepsis", title='Patient Prediction' )
  
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
            return render_template('patientprediction.html', upload_text="Uploaded Successfully",title='Patient Prediction' )
        else:
            return render_template('patientprediction.html', upload_text="Upload failed",title='Patient Prediction' )
        
   

if __name__=='__main__':
	app.run(debug=True)
