# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 23:22:12 2022

@author: Zoro
"""
import pandas as pd
import numpy as np
import nltk 
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
import re
import string
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import pickle
from pickle import load
from sklearn.feature_extraction.text import TfidfVectorizer

pickle_in = open("svm2.pkl","rb")
model=pickle.load(pickle_in)
cikle= open("tfid1.pkl","rb")
rating=pickle.load(cikle)

def clean_string(text):
    final_string = []
    text = text.lower()
    text = re.sub(r'\n', '', text)
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text=word_tokenize(text)
    useless_words = nltk.corpus.stopwords.words("english")
    useless_words = useless_words + ['hi', 'im']
    text_filtered = [word for word in text if not word in useless_words]
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]
    lem = WordNetLemmatizer()
    text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    final_string = ' '.join(text_stemmed)
    df=rating.transform([final_string])
    da=model.predict(df)
    return da


import streamlit as st
def main():
    st.title("Hotel Review Predictions") 
    html_temp = """
    <div style="background-color:grey;padding:2px">
    <h2 style="color:white;text-align:center;">SVM Model</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    test = st.text_input("Please enter a Review:\n")
 
    result=""
    if st.button("Predict"):
       result=clean_string(test)
       st.success('Review is  {}'.format(result))
if __name__=='__main__':
    main()

















 
