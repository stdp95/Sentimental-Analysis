# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 13:13:41 2017

@author: Satya
"""

import os
from collections import Counter
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.model_selection import train_test_split



def create_dictionary(path):
    #print os.listdir(path)
    #filepaths=glob(os.path.join(path,'*.txt'))
    all_words = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            with open(os.path.join(subdir, file)) as f:
                for line in f:
                    words = line.split()
                    all_words += words
    dictionary = Counter(all_words)
                
    all_keys = dictionary.keys()
    for key in all_keys:
        if key.isalpha() == False:
            del dictionary[key]
        if len(key)==1:
            del dictionary[key]
    dictionary=dictionary.most_common(3000)          
    return dictionary


def feature_extraction(path):
    #filepaths = glob(os.path.join(path,'*.txt')) 
    feature_matrix = np.zeros([2000,3000])
    for subdir, dirs, files in os.walk(path):
        for docId,filep in enumerate(files):    
            #print filep
            
            file_words=[]
            with open(os.path.join(subdir, filep)) as f:
                for line in f:
                    words = line.split()
                    file_words += words
        
            file_dict=Counter(file_words)
            
            for item in file_dict:
                for wordId,key in enumerate(dictionary):                
                    if key[0] == item:
                        feature_matrix[docId,wordId] =  file_dict[item]   
    
    return feature_matrix

root_path = 'C:\\Users\\Satya\\Desktop\\review_polarity\\txt_sentoken'
#dictionary=create_dictionary(root_path)

#feature_matrix=feature_extraction(root_path)
#np.save("movie_review_features.npy",feature_matrix)
feature_matrix = np.load("movie_review_features.npy")
#print feature_matrix
label = np.zeros(2000)
label[1001:] = 1
x_train , x_test , y_train , y_test=train_test_split(feature_matrix,label)    
     
#for train_index, test_index in skf.split(feature_matrix,label):
#    x_train , x_test = feature_matrix[train_index] , feature_matrix[test_index]
#    y_train , y_test = label[train_index] , label[test_index]
           
           
    
svm_model = LinearSVC()
svm_model.fit(x_train,y_train)     
    
gaussiannb = GaussianNB()
gaussiannb.fit(x_train,y_train)
    
multinomial_nb = MultinomialNB()
multinomial_nb.fit(x_train,y_train)
    
bernoulli_nb = BernoulliNB()
bernoulli_nb.fit(x_train,y_train)
    
nu_svc = NuSVC()
nu_svc.fit(x_train,y_train)
    
svc = SVC()
svc.fit(x_train,y_train)
    
    
predicted_label1 = svm_model.predict(x_test)       
predicted_label2 = gaussiannb.predict(x_test)         
predicted_label3 = multinomial_nb.predict(x_test)         
predicted_label4 = bernoulli_nb.predict(x_test)
predicted_label5 = svc.predict(x_test)
predicted_label6 = nu_svc.predict(x_test)
    
    
print (sum(y_test == predicted_label1))
print (sum(y_test == predicted_label2))
print (sum(y_test == predicted_label3))
print (sum(y_test == predicted_label4))
print (sum(y_test == predicted_label5))
print (sum(y_test == predicted_label6))
    
print (confusion_matrix(y_test,predicted_label1))
print (confusion_matrix(y_test,predicted_label2))
print (confusion_matrix(y_test,predicted_label3))
print (confusion_matrix(y_test,predicted_label4))
print (confusion_matrix(y_test,predicted_label5))
print (confusion_matrix(y_test,predicted_label6))
    