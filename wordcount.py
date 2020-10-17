import multidict as multidict
import bs4
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import os
from os import path
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
import unicodedata
import spacy
from spacy import tokenizer
nlp = spacy.load("en_core_web_sm")
import nltk
from nltk import tokenize
from nt import remove
from PIL import Image
# Created by Marco Saenz COO - BEPC INC. 
# Get Path Directory Local file within the same folder.
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
# Load in the dataframe *.csv file with words withihn the same folder. (In this example we are using a short description in FDA Audits observations for Medical Device companies for the past 12 years over 40K rows with 5 words in average per row)
text = open(path.join(d,'Layer2FDA.csv')).read()
#Remove Stopwords --- Required to avoid "is, and, or, etc..."
stopwords = set(STOPWORDS)
#Additional words you want to avoid?---->
stopwords.add("lack")
#Clean Data
# remove extra whitespace and break lines
text = re.sub(r'.*:', '', text)
text = re.sub('[\(\[].*?[\)\]]', ' ', text)
#Remove accented words.
def remove_accented_chars(text):
    unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text
#All Lowercase
text.lower()
#Function for removing Special Characters Gui can be found here: https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    re.sub(pattern, '', text)
    return text
# Function for Stemwords Jump --- Jumps-Jumped-Jumping...--->Jump
def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text
#Lemmatize text - Similar to Stem but precise for Dictionary and lexicographically correct words. for nlp function (natural languaje processing) it was needed to install the Spacy dictionary https://spacy.io/usage/models
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text
#Remove Stopwords unsing Tokenizer form Spacy and Tokenize from nltk words like "the, and, if, is" bringing these from teh STOPWORDS dictionary from wordcloud
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in STOPWORDS]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in STOPWORDS]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
#Load image form i.e. Circle-Star-Word- etc... I included 3 jpg in the folder you can pick what you like form there.
mask_d = np.array(Image.open(path.join(d, "Wordcloud_cloud.jpg")))
#Set Variable  with styles use WordCloud for questions and Appereance use WordClou? in python for insturctions.
wc = WordCloud(background_color='white',max_words=200, colormap= 'tab10', mask=mask_d, stopwords=STOPWORDS, min_word_length= 4, max_font_size=1250, relative_scaling=0.5)
#Generate the text that will be printed
wc.generate(text)
#Plot
plt.figure()
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
#While the wordcloud representation is not pretty it is accurate when calculating the repetitions of each word and clustering. Im currently working on a K-Means CLustering Method to see how that looks like for these clusters.
#You can find me on linkedin: https://www.linkedin.com/in/marco-saenz-pmp/
#Enjoy the Wordcloud for your Marketing efforts!

