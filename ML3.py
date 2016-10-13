from __future__ import division
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix
from numpy import loadtxt

def read_csv(csv_file,rows):
    data = pd.read_csv(csv_file,nrows = rows, iterator = True)
    labels = data['label']
    reviews = data['text']
    return labels,reviews

def get_ngrams_tf(reviews,n,is_binary_matrix,vocab):
    vectorizer = CountVectorizer(ngram_range=(n, n), binary = is_binary_matrix,vocabulary=vocab, token_pattern=r'\b\w+\b')
    ngrams_tf = vectorizer.fit_transform(reviews)
    return vectorizer.vocabulary_, ngrams_tf


def get_ngrams_idf(reviews_tf):
    log_factor = 0.434 #Log_e(10) for transorming natural log to log_10
    tfidf_transformer = TfidfTransformer()
    reviews_idf = tfidf_transformer.fit_transform(reviews_tf)
    return reviews_idf/log_factor

def calc_error_rate(result,testlabels):
    diff=result-testlabels
    errors=np.count_nonzero(diff)
    error_rate = np.divide(errors,len(testlabels))
    error_rate=(errors/len(testlabels))*100
    return error_rate

def naive_bayes(splits,labels,reviews_tf):
    kf = KFold(n_splits=splits)
    error_rates = []
    for train, test in kf.split(labels):
        train_data = uni_reviews_tf[train]
        test_data = uni_reviews_tf[test]
        train_labels = labels[train]
        test_labels = labels[test]
        bayes_classifier.fit(train_data, train_labels)
        predictions = bayes_classifier.predict(test_data)
        error_rate = calc_error_rate(predictions, test_labels)
        error_rates.append(error_rate)
    error_mean = np.mean(error_rates)
    print ("Bayes Errors: ")
    print error_rates
    print error_mean
    return error_mean,

def naive_bayes_mod(splits,labels,reviews):
    kf = KFold(n_splits=splits)
    error_rates = []
    for train, test in kf.split(labels):
        train_data = reviews[train]
        train_labels = labels[train]
        vocab_train,train_data_tf = get_ngrams_tf(train_data,1,True,None)
        test_data = reviews[test]
        vocab_test,test_data_tf = get_ngrams_tf(test_data,1,True,vocab_train)
        test_labels = labels[test]
        bayes_classifier.fit(train_data_tf, train_labels)
        predictions = bayes_classifier.predict(test_data_tf)
        error_rate = calc_error_rate(predictions, test_labels)
        error_rates.append(error_rate)
    error_mean = np.mean(error_rates)
    print ("Bayes Errors: ")
    print error_rates
    print error_mean
    return error_mean, error_rates



def k_fold(splits,):

    return

def perceptron_train(reviews,labels,max_iter):
    k_fold
    weights = csr_matrix(1,reviews[0].shape[1])
    averaged_weights = csr_matrix(1,reviews[0].shape[1])
    bias = 0
    averaged_bias = 0
    count = 1
    for i in range(reviews.shape[0]):
        print weights.shape
    return

labels,reviews = read_csv('reviews_tr.csv', 20000)
#UniGrams
uni_vocabulary_dict,uni_reviews_tf = get_ngrams_tf(reviews,1,True,None)
uni_reviews_idf = get_ngrams_idf(uni_reviews_tf)
bayes_classifier = BernoulliNB()
bayes_classifier.fit(uni_reviews_tf,labels)
bayes_error_mean,bayes_errors = naive_bayes_mod(5,labels,reviews)
exit(0)
modified_labels = labels+ labels - 1
print modified_labels




#BiGrams
bi_vocabulary_dict,bi_reviews_tf = get_ngrams_tf(reviews,1)
bi_reviews_idf = get_ngrams_idf(uni_reviews_tf)
print bi_reviews_idf.shape
print bi_vocabulary_dict

perceptron_train(uni_reviews_tf,labels,2)


#print vocabulary_dict
#out = ngrams(str1,3)
#print s
#print out
#labels = csv_f[0]
