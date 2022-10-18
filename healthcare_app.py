
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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

app = Flask(__name__)

model = pickle.load(open('sepsis_rf_sample.pkl', 'rb'))
UPLOAD_FOLDER = 'static/files'
UPLOAD_FILE = 'static/files/patientdata.csv'
NEW_Upload = 'static/files/train.csv'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/patientprediction')
def patientprediction():
    return render_template('patientprediction.html')
    
    
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        data = pd.read_csv(UPLOAD_FILE)
        patid = int(request.form['patid'])
        x = data.loc[data['PatientID'] == patid]
        #x.drop('PatientID', inplace=True, axis=1)
        x = x.iloc[:, 1:]
        
        inp = np.array(x).reshape((1, -1))
        prediction = model.predict(inp)
        output="no"
        if prediction[0]==1:
            output=""
     
        return render_template('patientprediction.html', prediction_text="Patient " +str(patid)+ " has " + output+" symtoms "+"of Sepsis" )
  
@app.route('/uploadfiles',methods=['POST'])
def uploadfiles():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
              # set the file path
            #uploaded_file.save(file_path)
              # save the file
            return render_template('patientprediction.html', upload_text="Uploaded Successfully" )
        else:
            return render_template('patientprediction.html', upload_text="Upload failed" )
        

@app.route('/uploadtraindata',methods=['POST'])
def uploadtraindata():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
              # set the file path
            # uploaded_file.save(file_path)
              # save the file
            return render_template('datatraining.html', upload_text="Uploaded Successfully" )
        else:
            return render_template('datatraining.html', upload_text="Upload failed" )





@app.route('/datatraining')
def datatrain():
    return render_template('datatraining.html')
    
def feature_selection(data):
    features = pd.read_csv('selected_features.csv')
    features = features['0'].to_list()
    df = data[features]
    return df
    
def balanceData(data):
    count_1 = data['SepsisLabel'].value_counts()[1]
    count_0 = data['SepsisLabel'].value_counts()[0]
    df_shuffled = data.sample(frac=1,random_state=7)
    if count_1 < count_0:     
        df_shuffled_0 = df_shuffled.loc[df_shuffled['SepsisLabel'] == 0].sample(n=count_1,random_state=42)
        df_shuffled_1 = df_shuffled.loc[df_shuffled['SepsisLabel'] == 1]
        
    else:
        df_shuffled_0 = df_shuffled.loc[df_shuffled['SepsisLabel'] == 1].sample(n=count_1,random_state=42)
        df_shuffled_1 = df_shuffled.loc[df_shuffled['SepsisLabel'] == 0]
        
    new_data = pd.concat([df_shuffled_1, df_shuffled_0])
    return new_data

def ConvertToCateg(data):
    data.loc[(data['Age'] >=10) & (data['Age'] <60), 'age'] = 'adult'
    data.loc[data['Age'] > 60, 'age'] = 'old'
    data.loc[(data['O2Sat'] >= 90) & (data['O2Sat'] <= 100), 'o2sat'] = 'normal'
    data.loc[(data['O2Sat'] < 90) & (data['O2Sat'] >= 0), 'o2sat'] = 'abnormal'
    data.loc[(data['HR'] >= 70) & (data['HR']<=110) & (data['Age'] < 10), 'hr'] = 'normal'
    data.loc[(data['HR'] >= 60) & (data['HR']<=100) & (data['Age'] > 10), 'hr'] = 'normal'
    data.loc[((data['HR'] < 70) | (data['Age'] >= 110)) & (data['Age']<10), 'hr'] = 'abnormal'
    data.loc[((data['HR'] < 60) | (data['HR'] >= 100)) & (data['Age'] >= 10), 'hr'] = 'abnormal'
    data.loc[(data['Temp'] >= 36) & (data['Temp'] <= 38),'temp'] = 'normal'
    data.loc[(data['Temp'] < 36) | (data['Temp'] > 38),'temp'] = 'abnormal'
    data.loc[(data['MAP'] >= 70) & (data['MAP'] < 100),'map'] = 'normal'
    data.loc[(data['MAP'] < 70) | (data['MAP'] >= 100),'map'] = 'abnormal'
    data.loc[(data['Resp'].between(30, 60)) & (data['Age'] < 1), 'Resp'] = 'normal'
    data.loc[(data['Resp'].between(24, 40)) & (data['Age'].between(1, 3)), 'resp'] = 'normal'
    data.loc[(data['Resp'].between(22, 34)) & (data['Age'].between(3, 6)), 'resp'] = 'normal'
    data.loc[(data['Resp'].between(18, 30)) & (data['Age'].between(6, 12)), 'resp'] = 'normal'
    data.loc[(data['Resp'].between(12, 16)) & (data['Age'].between(12, 18)), 'resp'] = 'normal'  
    data.loc[(data['Resp'].between(12, 20)) & (data['Age'] > 18), 'resp'] = 'normal'    
    data.loc[((data['Resp'] < 30) | (data['Resp'] > 60)) & (data['Age'] <1) ,'resp'] = 'abnormal'   
    data.loc[((data['Resp'] < 24) | (data['Resp'] > 40)) & (data['Age'].between(1, 3)) ,'resp'] = 'abnormal'   
    data.loc[((data['Resp'] < 22) | (data['Resp'] > 34)) & (data['Age'].between(3, 6)) ,'resp'] = 'abnormal'
    data.loc[((data['Resp'] < 18) | (data['Resp'] > 30)) & (data['Age'].between(6, 12)) ,'resp'] = 'abnormal'
    data.loc[((data['Resp'] < 12) | (data['Resp'] > 16)) & (data['Age'].between(12, 18)) ,'resp'] = 'abnormal'
    data.loc[((data['Resp'] < 12) | (data['Resp'] > 20)) & (data['Age'] > 18) ,'resp'] = 'abnormal'  
    data.loc[(data['FiO2'] < 0.8 ) ,'fio2'] = 'normal'
    data.loc[(data['FiO2'] >= 0.8 ),'fio2'] = 'abnormal'
    data.drop(labels=['Age','O2Sat', 'HR', 'Temp', 'MAP', 'Resp', 'FiO2'], axis=1, inplace=True)
    return data
 

le = LabelEncoder() 
def encode(data):
    for col in data.columns.values:
    # Encoding only categorical variables
        if data[col].dtypes=='object':
        # Using whole data to form an exhaustive list of levels
            data=data[col]
            le.fit(data.values)
            data[col]=le.transform(data[col])
    return data
rfc_pipe = Pipeline([
    ('rfc', RandomForestClassifier(max_depth=10, min_samples_leaf=3, min_samples_split=4,
                       n_estimators=200))])  

@app.route('/train',methods=['POST'])
def train():
    if request.method == 'POST':
        data = pd.read_csv(NEW_Upload)
        data = feature_selection(data)
        data = balanceData(data)
        data = data.ffill()
        data = data.bfill()
        data = ConvertToCateg(data)
            #data = encode(data)
    
            #data = data.fillna('Missing', inplace=True)
        #datanew = encode(data)
        data = data.apply(le.fit_transform)
        X_train, X_test, y_train, y_test = train_test_split(
        data.drop('SepsisLabel', axis=1),  # predictors
        data['SepsisLabel'],  # target
        test_size=0.25,  # percentage of obs in test set
        random_state=0)
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        rfc_pipe.fit(X_train, y_train)
        pred = rfc_pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        date = str(datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p"))
        file = open('sepsis_rf1.pkl', 'wb')
        pickle.dump(rfc_pipe, file)   
            
            
        return render_template('datatraining.html', prediction_text="Data uploaded is trained and model is updated succesfully and accuracy is " + str(acc))

if __name__=='__main__':
	app.run(debug=True)
