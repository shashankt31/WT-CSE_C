from flask import Flask, jsonify, request,render_template,url_for,request
import pandas as pd
import matplotlib.pyplot as plt
#importing os module
import os
#Parsing emails so that we can extract data, we will import email and email.policy modules
import email,email.policy
#importing counter
from collections import Counter
#importing re module to work with regular expression
import re 
from html import unescape
import nltk
from nltk.corpus import stopwords
#Replace all url inside email with word 'URL'
#importing urlextract package that will help to find and extract URLs from given string.
import urlextract
from urlextract import URLExtract
import numpy as np
#importing the train_test_split from scikit-learn library
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
#importing scipy.sparse which provides functions to work with sparse data
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import zipfile
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pickle
import joblib

import flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')

#Now let's write a function that takes an email as input and returns its content as plain text, whatever its format is:
def mail_text(mail):
    html = None
    for data in mail.walk():
        cnt_typ = data.get_content_type()
        if not cnt_typ in ("text/plain", "text/html"):
            continue
        try:
            contnt = data.get_content()
        except: # in case of encoding issues
            contnt = str(data.get_payload())
        if cnt_typ == "text/plain":
            return contnt
        else:
            hhtml = contnt
    if html:
        return mail_text(data)

filename = 'pickle.pkl'
   
#Creating a function to find precision, recal, confucion matrix and f1 score
def cal_performance(y_prediction):
    precison = precision_score(y_yes, y_prediction)
    recal = recall_score(y_yes, y_prediction)
    confuson_matrix = confusion_matrix(y_yes, y_prediction)
    f1score = f1_score(y_yes, y_prediction)
    return {"Matrix" : confuson_matrix, "Precision" : precison, "Recall" :recal, "F1" : f1score}
   
    
class My_Class1(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, remove_punctuation=True,
                 remove_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        
        self.remove_punctuation = remove_punctuation
        self.remove_urls = remove_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
       
    def transform(self, X, y=None):
        urlextractor = urlextract.URLExtract()
        stemmer = nltk.PorterStemmer()
        processed = []
        for email in X:
            text = mail_text(email)     #email into plain text
            
            if self.replace_numbers:     #replacing all numbers in email with word NUMBER
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', str(text))#regexIStough!!
            if self.remove_punctuation:    #Removing punctuation
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            if self.remove_urls and urlextractor is not None:    #replacing all url with word URL
                all_url = list(set(urlextractor.find_urls(text)))
                all_url.sort(key=lambda url: len(url), reverse=True)
                for oneurl in all_url:
                    text = text.replace(oneurl, " URL ")
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:    #Stemming
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            processed.append(word_counts)
        return np.array(processed)  
    
    def fit(self, X, y=None):
        return self
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))  


#Vectorization
#sparse data which is mostly unused data
class my_class(BaseEstimator, TransformerMixin):
#Pipeline takes an estimator class which is inherited from sklearn.base.BaseEstimator.
#It includes fit and transform method.
#To apply transformation on an input we inherited class TransformerMixin
  #building an vocabulary of most common words
  def __init__(self, vsize =1000):
    self.vsize =vsize
  #Now we implement fit method of BaseEstimator class
  def fit(self,data,y=None):
    totalcnt=Counter()
    for wcnt in data:
      for w,cnt in wcnt.items():
        totalcnt[w]+=min(cnt, 10)
    #We will use most_common() mtehod which returns list of top n element from most common to least common
    maxoccur=totalcnt.most_common()[:self.vsize]
    self.maxoccur=maxoccur
    self.vocab={w: index +1 for index, (w, cnt) in enumerate(maxoccur)}
    return self   
  #Next we implement transform method of BaseEstimator class
  def transform(self, data, y=None):
    row=[];col=[];d=[]
    for r, wcnt in enumerate(data):
      for w,cnt in wcnt.items():
        row.append(r)
        col.append(self.vocab.get(w,0))
        d.append(cnt)
    return csr_matrix((d,(row,col)),shape=(len(data),self.vsize+1))


#Creating a class that extends BaseEstimator and TransformerMixin

class My_Class12(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, remove_punctuation=True,
                 remove_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        
        self.remove_punctuation = remove_punctuation
        self.remove_urls = remove_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
       
    def transform(self, X, y=None):
            processed = []
            #for email in X:
            #text = mail_text02(email)     #email into plain text
            text=X
            if self.replace_numbers:     #replacing all numbers in email with word NUMBER
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', str(text))#regexIStough!!
            if self.remove_punctuation:    #Removing punctuation
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            if self.remove_urls and urlextractor is not None:    #replacing all url with word URL
                all_url = list(set(urlextractor.find_urls(text)))
                all_url.sort(key=lambda url: len(url), reverse=True)
                for oneurl in all_url:
                    text = text.replace(oneurl, " URL ")
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:    #Stemming
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            processed.append(word_counts)
            return np.array(processed)  
    
    def fit(self, X, y=None):
        return self   
    
    '''
    clf = joblib.load('model.pkl')
    count_vect = joblib.load('count_vect.pkl')
    to_predict_list = request.form.to_dict()
    review_text = clean_text(to_predict_list['review_text'])
    pred = clf.predict(count_vect.transform([review_text]))
    if pred[0]:
        prediction = "Positive"
    else:
        prediction = "Negative"

    return jsonify({'prediction': prediction})'''


@app.route('/predict', methods=['POST'])
def predict():
    ham_mails=[]
    spam_emails=[]
    



    
    # stop scikit-learn's deprecation warnings
    #warnings.filterwarnings("ignore", category=FutureWarning)
    #Logistic Regression classifier model
    #random state refers to randomly shuffling the data
    #Common value is between 0 to 42
    logistic_reg_model = LogisticRegression(random_state=42)
    # Doing predictions using Logistic Regression
    if request.method == 'POST':
        message = request.form['review_text']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        pred = int(my_prediction[0])
        if pred==0:
            res="HAM"
        else:
            res="SPAM"


    
    return jsonify({'prediction': res})

























    #############################################################################################################################
    if pred[0]:
        prediction = "Positive"
    else:
        prediction = "Negative"

    return jsonify({'prediction': prediction})
    


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)
