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


#We create a method in which email messages are parsed
def parse_mail(myfile):
  #opening file using open() method and "rb" for read binary mode
  f=open(myfile,"rb")
  #email.parser provides API used to parse all the email data
  #Parser have an policy object which controls behaviour, we used default
  return email.parser.BytesParser(policy=email.policy.default).parse(f)


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
    
#Creating a function to find precision, recal, confucion matrix and f1 score
'''def cal_performance(y_prediction, y_yes=train_category):
    precison = precision_score(y_yes, y_prediction)
    recal = recall_score(y_yes, y_prediction)
    confuson_matrix = confusion_matrix(y_yes, y_prediction)
    f1score = f1_score(y_yes, y_prediction)
    return {"Matrix" : confuson_matrix, "Precision" : precison, "Recall" :recal, "F1" : f1score}
 '''   
    
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
    
    
    
#Extract files from zip into the folder ham and spam
#ham=!unzip /content/easy_ham.zip
with zipfile.ZipFile("easy_ham.zip","r") as temp:
    temp.extractall()
#spam=!unzip /content/spam.zip
with zipfile.ZipFile("spam.zip","r") as temp1:
    temp1.extractall()
#lisdir() will return list of all files in directory
ham_mail_filelist=os.listdir("easy_ham/")
spam_mail_filelist=os.listdir("spam/")
#list comprehension to add folder path to each file so that we can easily access file
#Joining the folder path with file
ham_email_filepath=["easy_ham/" + filename for filename in ham_mail_filelist]
spam_emails_filepath=["spam/" + filename for filename in spam_mail_filelist]

ham_mails=[parse_mail(filename) for filename in ham_email_filepath]
spam_emails=[parse_mail(filename) for filename in spam_emails_filepath]


#train test split technique splits the dataset to estimate the performance by making prediction on unseen data
#Dataset is splitted into 2 sub dataset, commonly 80% and 20% used
#First is 80% training dataset to fit the model
#2nd subset dataset is Test dataset to evalute the performance of model
X = np.array(ham_mails + spam_emails)
y = np.array([0] * len(ham_mails) + [1] * len(spam_emails))
#train_test_split() funtion takes entire dataset and splits into two sub set
#random_number will initialise the random number generator
train_text, test_text, train_category, test_category= train_test_split(X, y, test_size=0.2, random_state=42)

#Vectorize train data and test data
demo_train_vectors = train_text[:2]
train_wordcount = My_Class1().fit_transform(demo_train_vectors)
demo_test_vectors = test_text[:2]
test_wordcount=My_Class1().fit_transform(demo_test_vectors.ravel())

my_train_vect = my_class(vsize=5)
train_vectors = my_train_vect.fit_transform(train_wordcount)
train_vectors.toarray()
my_test_vect=my_class(vsize=5)
test_vectors = my_test_vect.fit_transform(test_wordcount)


#We remove mean and scaling to unit variance i.e. standardizing the features.
sc=StandardScaler(with_mean=False,with_std=False)
#StandardScaler(*, copy=True, with_mean=True, with_std=True)
train_vectors_stand =sc.fit_transform(train_vectors.todense())
test_vectors_stand =sc.fit_transform(test_vectors.todense())


#DOing on complete data as we have to perform tests
#pipeline to perform sequence of different transformations 
#finds set of features, generate new features, select only some good features) of raw dataset before applying final estimator
#complete pipeline
pre_processing = Pipeline([("email_to_word_count", My_Class1()),
                          ("wordcount_to_vector", my_class()),
                          ])
#Preparing train data
final_x = pre_processing.fit_transform(train_text)

# stop scikit-learn's deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#Logistic Regression classifier model
#random state refers to randomly shuffling the data
#Common value is between 0 to 42
logistic_reg_model = LogisticRegression(random_state=42)
# Doing predictions using Logistic Regression
y_prediction_logistic_reg  = cross_val_score(logistic_reg_model, final_x, train_category, cv=3, verbose=3)
y_prediction_logistic_reg.mean()
