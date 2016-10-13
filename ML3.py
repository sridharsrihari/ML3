
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from numpy import loadtxt

def ngrams(input, n):
  input = input.split(' ')
  output = {}
  for i in range(len(input)-n+1):
    g = ' '.join(input[i:i+n])
    output.setdefault(g, 0)
    output[g] += 1
  return output

def read_csv(csv_file):
    f = open(csv_file)
    csv_f = csv.reader(f)
    i = 0
    labels = []
    reviews = []
    for row in csv_f:
        if(i > 200):
            break
        if(i > 0):
            labels.append(row[0])
            reviews.append(row[1])
        i+=1
    return labels,reviews


def get_tf(reviews):
    count_vect = CountVectorizer()
    reviews_tf = count_vect.fit_transform(reviews)
    print reviews_tf.shape
    return count_vect.vocabulary_, reviews_tf


def get_idf(reviews_tf):
    tfidf_transformer = TfidfTransformer()
    reviews_idf = tfidf_transformer.fit_transform(reviews)
    print reviews_idf.shape
    return reviews_idf

def get_ngrams_tf(reviews,n):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    ngrams_tf = vectorizer.fit_transform(reviews)
    print ngrams_tf.shape
    return vectorizer.vocabulary_, ngrams_tf

def get_ngrams_idf(reviews_tf):
    tfidf_transformer = TfidfTransformer()
    reviews_idf = tfidf_transformer.fit_transform(reviews_tf)
    print reviews_idf.shape
    return reviews_idf




def perceptron_train(reviews,labels,max_iter):
    weights = np.zeros(reviews[0].shape)
    averaged_weights = np.zeros(reviews[0].shape)
    bias = 0
    averaged_bias = 0
    count = 1
    for i in range(max_iter):
        for j in range(reviews.shape[0]):
            return


    print weights.shape
    return

#str1 = ''.join(s)
labels,reviews = read_csv('reviews_tr.csv')
uni_vocabulary_dict,uni_reviews_tf = get_ngrams_tf(reviews,2)
uni_reviews_idf = get_ngrams_idf(uni_reviews_tf)
print uni_reviews_idf.shape
print uni_vocabulary_dict
exit(0)
vocabulary_dict, reviews_tf = get_tf(reviews)
reviews_idf = get_idf(reviews_tf)
print reviews_tf.shape
print reviews_idf.shape
perceptron_train(reviews_tf,labels,2)


#print vocabulary_dict
#out = ngrams(str1,3)
#print s
#print out
#labels = csv_f[0]
