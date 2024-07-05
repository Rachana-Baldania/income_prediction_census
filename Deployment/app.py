# Importing the necessary libraries
from flask import Flask, request, render_template
from flask_cors import CORS,cross_origin
import numpy as np
import pickle

app = Flask(__name__)  # Initialising flask app


@app.route('/', methods=['GET'])  # route to display the Home page
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in web UI
@cross_origin()
def prediction():
    if request.method == 'POST':
        # reading the inputs given by the user

        age = int(request.form['age'])
        hours = int(request.form['hours'])
        gain = int(request.form['gain'])
        loss = int(request.form['loss'])
        sex = request.form['sex']
        marital = request.form['marital']
        country = request.form['country']
        education = request.form['education']
        occupation = request.form['occupation']
        relationship = request.form['relationship']
        race = request.form['race']

        sex = 1 if sex == 'Male' else 0
        marital = 0 if marital == 'Married' else 1
        country = 1 if country == 'United States' else 0

        # if condition match only assign '1' to that variable
        _11th = 1 if education == '_11th' else 0
        _12th = 1 if education == '_12th' else 0
        _1st_4th = 1 if education == '_1st_4th' else 0
        _5th_6th = 1 if education == '_5th_6th' else 0
        _7th_8th = 1 if education == '_7th_8th' else 0
        _9th = 1 if education == '_9th' else 0
        _Assoc_acdm = 1 if education == 'Assoc_acdm' else 0
        assoc_voc = 1 if education == 'assoc_voc' else 0
        bachelors = 1 if education == 'bachelors' else 0
        doctorate = 1 if education == 'doctorate' else 0
        HS_grad = 1 if education == 'HS_grad' else 0
        masters = 1 if education == 'masters' else 0
        preschool = 1 if education == 'preschool' else 0
        prof_school = 1 if education == 'prof_school' else 0
        college = 1 if education == 'college' else 0
        
        Armed_Forces = 1 if occupation=='Armed-Forces' else 0
        Craft_repair = 1 if occupation=='Craft-repair' else 0
        Exec_managerial = 1 if occupation=='Exec-managerial' else 0
        Farming_fishing = 1 if occupation=='Farming-fishing' else 0
        Handlers_cleaners= 1 if occupation=='Handlers-cleaners' else 0
        Machine_op_inspct = 1 if occupation=='Machine-op-inspct' else 0
        service = 1 if occupation=='service' else 0
        Priv_house_serv = 1 if occupation=='Priv-house-serv' else 0
        Prof_specialty = 1 if occupation=='Prof-specialty' else 0
        Protective_serv = 1 if occupation=='Protective-serv' else 0
        Sales = 1 if occupation=='Sales' else 0
        Tech_support = 1 if occupation=='Tech-support' else 0
        Transport_moving= 1 if occupation=='Transport-moving' else 0
        
        Not_in_family= 1 if relationship=='Not-in-family' else 0
        Other_relative = 1 if relationship=='Other-relative' else 0
        Own_child = 1 if relationship=='Own-child' else 0
        Unmarried = 1 if relationship=='Unmarried' else 0
        Wife = 1 if relationship=='Wife' else 0
        
        Asian_Pac_Islander = 1 if race=='Asian-Pac-Islander' else 0
        Black = 1 if race=='Black' else 0
        Other= 1 if race=='Other' else 0
        White = 1 if race=='White' else 0

        # load the model
        model = pickle.load(open('/Users/rachanabaldania/Code/Employee-Income-Prediction/Deployment/xgb_model.pkl', 'rb'))
        # load the scaler
        scaler = pickle.load(open('/Users/rachanabaldania/Code/Employee-Income-Prediction/Deployment/scaler.pkl', 'rb'))

        # feature scaling on age,capital_gain, capital_loss, hours per week
        scaled_value = scaler.transform([[age, gain, loss, hours]])
        age, gain, loss, hours = scaled_value[0, 0], scaled_value[0, 1], scaled_value[0, 2], scaled_value[0, 3]

        # predictions using the loaded model file
        predict = model.predict(np.array([[age, hours, gain, loss, _11th, _12th, _1st_4th, _5th_6th, _7th_8th,_9th,_Assoc_acdm,
                  assoc_voc, bachelors,doctorate, HS_grad, masters, preschool, prof_school, college,
                  marital, sex, country,Armed_Forces,Craft_repair,Exec_managerial,Farming_fishing,Handlers_cleaners,Machine_op_inspct,service,Priv_house_serv,Prof_specialty,Protective_serv,Sales,Tech_support,Transport_moving,Not_in_family,Other_relative,Own_child,Unmarried,Wife,Asian_Pac_Islander,Black,Other,White]]))[0]
        output = "Annual Income is More Than 50K" if predict == 1 else "Annual Income is Less Than 50K"

        # showing the prediction result in a UI
        return render_template('result.html', output=output)


if __name__ == '__main__':
    app.run(debug=True)
