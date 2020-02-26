#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 12:24:21 2020

@author: czoppson
"""

import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('contest_tweets_2.csv_trainset.csv')

data.head()

tweets = data[['text','target']]

data.shape

corpus =[]

for i in range(data.shape[0]):
    corpus.append(data.iloc[i,1])
    
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())

X.shape

#X.toarray()

data['target'].plot(kind='hist')


###### cleaning part ##########
#import seaborn as sns
#import matplotlib.pyplot as plt
#from sklearn.feature_extraction.text import CountVectorizer
#import nltk 
#import string
#import re
#
#
#import nltk
#from nltk.corpus import stopwords
##nltk.download('stopwords')
#stop = stopwords.words('english')


tweets = data[['text','target']]

        
def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

tweets['Tweet_punct'] = data['text'].apply(lambda x: remove_punct(x))

def tokenization(text):
    text = re.split('\W+', text)
    return text

tweets['Tweet_tokenized'] = tweets['Tweet_punct'].apply(lambda x: tokenization(x.lower()))
tweets.head()

tweets.columns

stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
    
tweets['Tweet_nonstop'] = tweets['Tweet_tokenized'].apply(lambda x: remove_stopwords(x))

ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

tweets['Tweet_stemmed'] = tweets['Tweet_nonstop'].apply(lambda x: stemming(x))


wn = nltk.WordNetLemmatizer()
nltk.download('wordnet')

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

tweets['Tweet_lemmatized'] = tweets['Tweet_nonstop'].apply(lambda x: lemmatizer(x))




def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text


tweets.columns


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
import string
import re
from nltk.corpus import stopwords
#nltk.download('stopwords')
stop = stopwords.words('english')

ps = nltk.PorterStemmer()
stopword = nltk.corpus.stopwords.words('english')

def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text

countVectorizer = CountVectorizer(analyzer=clean_text) 
countVector = countVectorizer.fit_transform(tweets['text'])
print('{} Number of tweets has {} words'.format(countVector.shape[0], countVector.shape[1]))

countVector.shape # (32026,48504)


count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())
count_vect_df.head()



# TF-IDF TF is  number of times term t appears in a document / total number of terms in document 
# IDf is log(total number of documents/ number of documents with term t in it)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(countVector)
X_train_tfidf.shape

X_train_tfidf.toarray()


# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, data.target)
clf.score(X_train_tfidf, data.target)
clf.get_params

#####PIPELINE####

from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer(analyzer=clean_text,ngram_range = (1,2))), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB(alpha = 0.01))])
text_clf = text_clf.fit(data.text, data.target)

text_clf.score(data.text, data.target)#### 0.87169 # z alpha 0.01 bliskie 99 procent 

test_data = pd.read_csv("testset_notarget.csv")
predicted = text_clf.predict(test_data.text)

np.unique(predicted,return_counts = True) # 607, 8783,  573, 2461


panda  = {'id':np.arange(len(predicted)),'target':predicted}
df = pd.DataFrame(panda)

df.to_csv(r'/Users/czoppson/Desktop/Machine_learning/Kaggle_nlp/politicians-and-celebrities/predictions_normal_bayes99.csv',index = False)


# Training Support Vector Machines - SVM and calculating its performance

from sklearn.linear_model import SGDClassifier

text_clf_svm = Pipeline([('vect', CountVectorizer(analyzer=clean_text)), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42))])

text_clf_svm = text_clf_svm.fit(data.text, data.target)
text_clf_svm.score(data.text, data.target)#0.731

from sklearn.model_selection import GridSearchCV

parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf-svm__alpha': (1e-2, 1e-3),}

gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(data.text, data.target)
gs_clf_svm.best_score_
gs_clf_svm.best_params_
prediction_svm = gs_clf_svm.predict(test_data.text)
np.unique(prediction_svm,return_counts = True)

np.unique(data.target,return_counts = True)

import sklearn.svm

sigma =0.1
gamma = 1/(2*sigma**2)

text_clf_svm_ml = Pipeline([('vect', CountVectorizer(analyzer=clean_text)), ('tfidf', TfidfTransformer()),
                         ('clf-svm',sklearn.svm.SVC(C =10,kernel = 'rbf',gamma = 'scale'))])
    
text_clf_svm_ml = text_clf_svm_ml.fit(data.text, data.target)
text_clf_svm_ml.score(data.text, data.target)



# linear kernel
text_clf_svm_ml_linear = Pipeline([('vect', CountVectorizer(analyzer=clean_text)), ('tfidf', TfidfTransformer()),
                         ('clf-svm',sklearn.svm.SVC(C =10,kernel = 'linear',gamma = 'scale'))])
    
text_clf_svm_ml_linear = text_clf_svm_ml.fit(data.text, data.target)
score_svm_ml = text_clf_svm_ml_linear.score(data.text, data.target)
predicted = text_clf_svm_ml_linear.predict(test_data.text)
np.unique(predicted,return_counts = True)

panda  = {'id':np.arange(len(predicted)),'target':predicted}
df = pd.DataFrame(panda)

df.to_csv(r'/Users/czoppson/Desktop/Machine_learning/Kaggle_nlp/politicians-and-celebrities/predictions_svm_linear.csv',index = False)




####### GRIDSEARCH
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2),(2,2)],
              'tfidf__use_idf': (True, False),
              'tfidf__smooth_idf':(True,False),
              'tfidf__sublinear_tf':(True,False),
              'clf__alpha': (1, 1e-5),
              'clf__fit_prior':(True,False),}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(data.text, data.target)

gs_clf.best_score_
gs_clf.best_params_



#linki 
https://towardsdatascience.com/twitter-api-and-nlp-7a386758eb31 # trump vs hillary
https://www.kaggle.com/ragnisah/text-data-cleaning-tweets-analysis # data cleaning 
https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a # pipline and text calssification also cvGRID 
https://towardsdatascience.com/a-simple-example-of-pipeline-in-machine-learning-with-scikit-learn-e726ffbb6976 # simple example of a pipline
https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5 # different classifications
https://medium.com/datadriveninvestor/an-introduction-to-grid-search-ff57adcc0998 # introduction to GRIDsearch CV
https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f # kozackie porównanie róznych metod
https://github.com/miguelfzafra/Latest-News-Classifier/blob/master/0.%20Latest%20News%20Classifier/04.%20Model%20Training/10.%20MT%20-%20Multinomial%20LogReg.ipynb # github typka co ma o tym mnóstwo info  



from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

text_clf_log_reg = Pipeline([('vect', CountVectorizer(analyzer=clean_text)), ('tfidf', TfidfTransformer()),
                         ('clf',OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1))])
text_clf_log_reg = text_clf_log_reg.fit(data.text,data.target)
score_log_reg = text_clf_log_reg.score(data.text,data.target)# 90

parameters_log_reg = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'tfidf__smooth_idf':(True,False),
              'tfidf__sublinear_tf':(True,False),
              'clf__C' : np.logspace(-3,3,7),
              'clf__penalty': [('l1'),('l2')],
              }

gs_log_reg = GridSearchCV(text_clf_log_reg, parameters_log_reg, n_jobs=-1)
gs_log_reg = gs_log_reg.fit(data.text,data.target)



##### real logre


text_clf_log_reg2 = Pipeline([('vect', CountVectorizer(analyzer=clean_text,ngram_range = (1,2))), ('tfidf', TfidfTransformer()),
                         ('clf',LogisticRegression())])#solver='sag'
text_clf_log_reg2 = text_clf_log_reg2.fit(data.text,data.target)
score_log_reg2 = text_clf_log_reg2.score(data.text,data.target) # 90 

predicted_log = text_clf_log_reg2.predict(test_data.text)

panda  = {'id':np.arange(len(predicted)),'target':predicted_log}
df = pd.DataFrame(panda)

df.to_csv(r'/Users/czoppson/Desktop/Machine_learning/Kaggle_nlp/politicians-and-celebrities/predictions_reg_log.csv',index = False)





np.unique(predicted,return_counts = True)
np.unique(predicted_log,return_counts = True)

sum(predicted == predicted_log)

parameters_log_reg2 = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'tfidf__smooth_idf':(True,False),
              #'clf__C' : np.logspace(-3,3,7),
              'clf__penalty': [('l1'),('l2')],
              }

gs_log_reg2 = GridSearchCV(text_clf_log_reg2, parameters_log_reg2, n_jobs=-1)
gs_log_reg2 = gs_log_reg2.fit(data.text,data.target)

gs_log_reg2.best_score_

##################################################################

from random import sample 


np.unique(data.target,return_counts = True)

celebrities_idx = data.index[data.target =='celebrity']
biz_idx = data.index[data.target =='biz&tech']
internet_idx = data.index[data.target =='internetplatform']
ploitician_idx = data.index[data.target =='politician']

#data_reduced 

my_list = sample(list(celebrities_idx),2700)
sample(list(biz_idx),2700)
sample(list(internet_idx),2700)
sample(list(ploitician_idx),2700)

my_list.extend(sample(list(ploitician_idx),2700))
len(my_list)/4

data_reduced = data.iloc[my_list]

np.unique(data_reduced.target,return_counts = True)


text_clf = Pipeline([('vect', CountVectorizer(analyzer=clean_text,ngram_range = (1,2))), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB(alpha = 0.01))])
text_clf = text_clf.fit(data_reduced.text, data_reduced.target)

text_clf.score(data_reduced.text, data_reduced.target)#### 0.87169 # z alpha 0.01 bliskie 99 procent 

test_data = pd.read_csv("testset_notarget.csv")
predicted = text_clf.predict(test_data.text)

np.unique(predicted,return_counts = True) # 607, 8783,  573, 2461


panda  = {'id':np.arange(len(predicted)),'target':predicted}
df = pd.DataFrame(panda)

df.to_csv(r'/Users/czoppson/Desktop/Machine_learning/Kaggle_nlp/politicians-and-celebrities/predictions_reduced_data2.csv',index = False)

df.shape