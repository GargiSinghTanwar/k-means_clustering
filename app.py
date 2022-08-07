import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
classifier = pickle.load(open('cluster.pkl','rb')) 


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    Gener = int(request.args.get('Gener'))
    age = int(request.args.get('age'))
    Annual_income = int(request.args.get('Annual_income'))
    Spending_score = int(request.args.get('Spending_score'))
    prediction = classifier.predict([[Gener,age,Annual_income,Spending_score]])
    
    print("K-means prediction",prediction)
    if prediction==[0]:
      print("cluster 0")
    elif prediction==[1]:
      print("cluster 1")
    else:
      print("cluster 2")

        
    return render_template('index.html', prediction_text='k-means cluster = {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
