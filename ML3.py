
import csv

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
        if(i>0):
            labels.append(row[0])
            reviews.append(row[1])
        i+=1
    return labels,reviews


#str1 = ''.join(s)
labels,reviews = read_csv('reviews_tr.csv')
print reviews[2]
#out = ngrams(str1,3)
#print s
#print out
#labels = csv_f[0]
