#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 12:01:02 2021

@author: bushrausif
"""

import pandas as pd
import numpy as np
import nltk
from flask import Flask,render_template,request
import pickle
import re 
import flask
from nltk import WordNetLemmatizer, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__,template_folder='template') #Initialize the flask App

model = pickle.load(open('modelDs.pkl', 'rb')) # loading the trained model
vectorizer = pickle.load(open("vectorizerds.pkl", "rb"))

@app.route('/') # Homepage
def home():
    return render_template('home.html')


stop_words=set(nltk.corpus.stopwords.words('english'))
mystopword=['resource','general','at','read','important','respect','single',
           'term','images','believe','besides','anyone','many','beginners','peresent','overall','chapter',
           'people','class','hands','someone','ideas','thinking','how to','two','my job','opened','even','uh','graduate',
            'student','social', 'psychology','transition','into','side','job','examples','example','classes','somewhat',
            'person','seriously','excel','course','teach','match','users','tool','started','my mind','mind','my openins'
           ,'project','time','spent','anyway','design','powerpoint','career','field','point','fields','following',
           'working','chapters','freshman','college-freshman','retains','matter','whether'
           ,'however','leading','starting','write','choices','whatsoever','knowlegde','name','approach','guy'
           'writing','download','second','makes','collection','handbook','interviews','interview','answer'
           ,'consume','econometrics','found','provided','appendix','table','table a','appendix a','impression'
           ,'background','research','modelling','hang on','among','words','sentences','even','paragraphs'
            ,'introduction','ways','designed','probably','induces','son','my son','newborn','function',
            'refreshing','installation','newbie','walk','environment','sections','works','archetype','bodes','code','everyone'
            ,'dummies','numerous','factors','models','network','consider','talking','let','say','joke','calculus','education',
           'algebra','subject','stat','dealing','doubly','segmentation','ph','alot','ago','tldr','alongside','equation',
            'tour','crumb','thing','applied','ga','sw','guide','guile','trying','attack','touch','bread','various'
            ,'image','ai','leaf','statistic','outset','human','plan','math','degree','true','darned','joke'
           ,'book','lgbtq','account','auther','thought','always','included','amazon','already','although','page','although','attribute',
            'area','wheelan','usually','woman','actually','repeat','year','also','daughter','machine','zumel',
            'zorach','zork','zuboff','zuck','zuckerberg','zuse','zynga','nan','textbook','make','need','seem','auther'
           ,'author','good','first','last','work','find','think','possible','password','email','much','book','read','whole'
            ,'seems','like','liked','device','system','computer','kids','chileds','books','reading','write','reader','take','look'
            ,'long','used','well'

           ]
setstop = set(mystopword) 
stopls = list(setstop) 

def clean_text(text):
    # remove all word contain uneccassry and integer
    
    text = re.sub("]", "", text)
    #remove URLs
    text = re.sub(r"http\S+", "", text,flags=re.U)    

    # Remove user @ references and '#' from tweet
    text = re.sub(r'\@\w+|\#\w+','', text)
    text = re.sub("\n", " ", text)
    text = re.sub(' +', ' ', text)  # removing unnecessary spaces
    text= re.sub('\w*\d\w*','',text)
    

    return text.strip()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
 #   def clean_text(headline):
    le=WordNetLemmatizer()
    word_tokens=word_tokenize(text)
    stop_words=set(nltk.corpus.stopwords.words('english'))
    tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
    cleaned_text=" ".join(tokens)
    return cleaned_text
def remove_repeated_letters(text):
    
    text = re.sub(r'(.)\1+', r'\1\1', text)  
   # text = re.sub(r'(.)\1+', r'\1', text) 

    return text

def remove_stopwords(text):
    stop_words=set(nltk.corpus.stopwords.words('english'))
    text_list = text.split(' ')  # to remove stopwords easily  
    new_text_list = []
    for s in text_list:
        if not (s in stop_words) and not(s in stopls): #Checks two lists
            new_text_list.append(s)
        
    text = ' '.join(e for e in new_text_list)  # gather text together again

    return text
    
def preprocess_text(text):
    # clean text
    text = clean_text(text)
  
    # remove repeated letters
    text = remove_repeated_letters(text)

    # remove stopwords
    text = remove_stopwords(text)
    #text=only_english(text)
    

    return text


@app.route('/predict',methods=['POST'])

def predict():
    if request.method == 'POST':
        usr_input = str(request.form.get('usr'))
        comment=preprocess_text(usr_input)
        print(comment)
        data = [comment]
        vect = vectorizer.transform(data).toarray()
        label = model.predict(vect)
        print(label)
    
        
        return render_template('home.html',val=label[0])
    return render_template('home.html')

 
if __name__ == "__main__":
    app.run(debug=True)
    