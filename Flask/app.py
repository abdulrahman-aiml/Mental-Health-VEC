# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:52:54 2024

@author: indra
"""

import numpy as np
import pickle
import pandas
import os
from flask import Flask, request, render_template
import re
import nltk
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#nltk.download('punkt')

app = Flask(__name__)
model = pickle.load(open(r'Flask\model.pkl', 'rb'))
tfidf_vectorizer = joblib.load(open(r'Flask\tf.pkl', 'rb'))


@app.route('/') # rendering the html template
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST", "GET"]) # rendering the html template
def predict() :
    return render_template("input.html")

@app.route('/submit',methods=["POST","GET"])# route to show the predictions in a web UI
def submit():
    #  reading the inputs given by the user
    text = request.form['userInput']
    text=re.sub('[^a-zA-Z0-9]+'," ",text)
    
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    preprocess_text = ' '.join(lemmatized_tokens)

    print(preprocess_text)
    # Transform the preprocessed text using the TF-IDF vectorizer
    text_vectorized = tfidf_vectorizer.transform([preprocess_text])
    
    # Make predictions using the model
    prediction = model.predict(text_vectorized)[0]
    
    # Map prediction to label
    label = "Positive" if prediction == 1 else "Negative"
    
    return render_template('output.html', prediction_text=f'After Analysis State of Mind was found: {label}')


if __name__=="__main__":
    
    # app.run(host='0.0.0.0', port=8000,debug=True)    # running the app
    port=int(os.environ.get('PORT',5000))
    app.run(debug=True)