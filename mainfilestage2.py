import os
import random
import sys
from dataclasses import replace

import nltk
import gensim
import numpy as np
import csv
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from numpy import array
import ast
import re
import math
import json
import difflib
from utilities import removestopwords, stemming, normalize_data, rmse, clean_dataset, splits, printresults, output_result_files, rq2, rq3, rq1

from approach4 import run_entity_similarity, run_relationship_similarity, approach4
from approach1 import approach1
from approach2 import run as approach2
from approach3 import approach3
from approach5 import approach5
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

#edit training dataset paths here (known grades)
ds1path = "C:/Users/learj/PycharmProjects/pythonProject/cs473/project/for_students/for_students/Dataset1"
ds2path = "C:/Users/learj/PycharmProjects/pythonProject/cs473/project/for_students/for_students/Dataset2"
#edit testing dataset paths here (unseen grades)
ds1tpath = "C:/Users/learj/PycharmProjects/pythonProject/cs473/project/for_testing/for_testing/Dataset1"
ds2tpath = "C:/Users/learj/PycharmProjects/pythonProject/cs473/project/for_testing/for_testing/Dataset2"

#edit path to the training dataset grades here
gradespath = "C:/Users/learj/PycharmProjects/pythonProject/cs473/project/for_students/for_students/ERD_grades.csv"
f = open(gradespath, "r")
ds1grades = []
ds2grades = []
count = 0
for line in f:
    parts = line.split(",")
    if count != 0:
        ds1grades.append((parts[0], float(parts[1])))
        ds2grades.append((parts[0], float(parts[2])))
    count += 1

dataset1, itemcounts1 = clean_dataset(ds1path)
dataset2, itemcounts2 = clean_dataset(ds2path)
dataset1t, itemcounts1t = clean_dataset(ds1tpath)
dataset2t, itemcounts2t = clean_dataset(ds2tpath)


dataset1, collectionlist1 = normalize_data(dataset1)
dataset2, collectionlist2 = normalize_data(dataset2)
dataset1t, collectionlist1t = normalize_data(dataset1t)
dataset2t, collectionlist2t = normalize_data(dataset2t)

#TO GET RESULTS OF ACTUAL TESTING DATA, SET BOTH DATASET SPLITS and PRINT RESULTS TO 'testing = False'
testing = False
seed = 500
trainingdata1, testingdata1, trainingdatanames1, testingdatanames1, trainingdataconts1, testingdataconts1, trainitemcount1, testitemcount1 = splits(dataset1, collectionlist1, ds1grades, itemcounts1, dataset1t, itemcounts1t, collectionlist1t, testing = testing, seed=seed)
trainingdata2, testingdata2, trainingdatanames2, testingdatanames2, trainingdataconts2, testingdataconts2, trainitemcount2, testitemcount2 = splits(dataset2, collectionlist2, ds2grades, itemcounts2, dataset2t, itemcounts2t, collectionlist2t, testing = testing, seed=seed)

#k=4


#HERE WE GET RESULTS FOR APPROACH 1
#params1 = {'A': 1.3817379916687071, 'B': 3.8896820445671914, 'C': 3.322315614224125, 'D': 1.4414797003609219, 'alpha': 0.9047016478956881, 'beta': .5, 'K': 6}
#version of params that factors in comments
params1 = {'A': 2.8573974765927592, 'B': 0.6712578444597177, 'C': 3.601158590055459, 'D': 3.000557922549613, 'alpha': 0.9991572245411128, 'beta': 0.035051644786343844, 'K': 5}

k = params1["K"]
results11 = approach1(trainingdata1, testingdata1, trainingdataconts1, testingdataconts1, trainingdatanames1, testingdatanames1, trainitemcount1, testitemcount1, ds1grades, params1, k=k)
results12 = approach1(trainingdata2, testingdata2, trainingdataconts2, testingdataconts2, trainingdatanames2, testingdatanames2, trainitemcount2, testitemcount2, ds2grades, params1, k=k)


#HERE WE GET RESULTS FOR APPROACH 2
if testing is False:
    results21 = approach2(gradespath, ds1path, seed=seed, testing=ds1tpath)
    results22 = approach2(gradespath, ds2path, seed=seed, testing=ds2tpath)
else:
    results21 = approach2(gradespath, ds1path, seed=seed)
    results22 = approach2(gradespath, ds2path, seed=seed)

#HERE WE GET RESULTS FOR APPROACH 3
results31 = approach3(trainingdata1, testingdata1, trainingdataconts1, testingdataconts1, testingdatanames1, gradespath)
results32 = approach3(trainingdata2, testingdata2, trainingdataconts2, testingdataconts2, testingdatanames2, gradespath)


#HERE WE GET RESULTS FOR APPROACH 4

#different params based on different training
params = {'A': 0.008097658971528523, 'B': 0.013945646059937452, 'C': 0.21883992806812, 'D': 0.7591167669004141, 'T': 0.06597323666238664, 'U': 0.00762455521148119, 'V': 0.0048008844641487525, 'X': 0.012301884649412755, 'Y': 0.024146228140775598, 'Z': 0.8851532108717951, 'alpha': 0.006497542159462399, 'beta': 0.10882968354314841, 'lambda': 0.8846727742973891, 'entity_type_c': 0.5743956407590298, 'rel_type_c': 0.40550007128340027, 'K': 8}
params2 = {'A': 0.10943132873974656, 'B': 0.7610961401053422, 'C': 0.021271558352512095, 'D': 0.10820097280239913, 'T': 0.020784881396816998, 'U': 0.487607060588602, 'V': 0.487607060588602, 'X': 0.003416098645794017, 'Y': 0.0005109664328962965, 'Z': 7.393234728857425e-05, 'alpha': 0.007249927562086892, 'beta': 0.008487761526945596, 'lambda': 0.9842623109109676, 'entity_type_c': 0.39251387917953096, 'rel_type_c': 0.11398878584290015, 'K': 7}

k = params['K']
results41 = approach4(dataset1, collectionlist1, testingdata1, trainingdata1, testingdataconts1,trainingdataconts1, trainingdatanames1, testingdatanames1, trainitemcount1, testitemcount1, ds1grades, params, k=k)
results42 = approach4(dataset2, collectionlist2, testingdata2, trainingdata2, testingdataconts2,trainingdataconts2, trainingdatanames2, testingdatanames2, trainitemcount2, testitemcount2, ds2grades, params, k=k)


#HERE WE GET RESULTS FOR APPROACH 5
# #params based on tuning (5.3)
#params5 = {'A': 0.42120146018464794, 'B': 0.03237487568431767, 'C': 0.0011256476086930795, 'D': 0.5452980165223413}
#params based on softmax of RMSE (current 5.2)
#params5 = {'A': 0.319870883833914, 'B': 0.001165857600862784, 'C': 0.2356521555471026, 'D': 0.44331110301812066}
#test params
#params5 = {'A': 0.3202442430187699, 'B': 0, 'C': 0.23592721308172426, 'D': 0.44382854389950577}
params5 = {'A': 0.34223158773916196, 'B': 0.02463664377658566, 'C': 0.21974512480093306, 'D': 0.4133866436833193}
results51 = approach5(results11, results21, results31, results41, ds1grades, params5)
results52 = approach5(results12, results22, results32, results42, ds2grades, params5)


# testsetnameranges = max(len(testingdatanames1), len(testingdatanames2))
# if testsetnameranges == len(testingdatanames1):
#     testsetnames = testingdatanames1
# else:
#     testsetnames = testingdatanames2
testsetnames = []
for t in testingdatanames1:
    if t not in testsetnames:
        testsetnames.append(t)
for t in testingdatanames2:
    if t not in testsetnames:
        testsetnames.append(t)
testsetnames = sorted(testsetnames)

file1 = printresults(results11, results12, ds1grades, ds2grades, testsetnames, testing = testing)
file2 = printresults(results21, results22, ds1grades, ds2grades, testsetnames, testing = testing)
file3 = printresults(results31, results32, ds1grades, ds2grades, testsetnames, testing = testing)
file4 = printresults(results41, results42, ds1grades, ds2grades, testsetnames, testing = testing)
file5 = printresults(results51, results52, ds1grades, ds2grades, testsetnames, testing = testing)

#
output_result_files(file1=file1, file2=file2, file3=file3, file4=file4, file5=file5)
#ap1rmse = (rmse(ds1grades, results11)[0], rmse(ds2grades, results12)[0])
#ap2rmse = (rmse(ds1grades, results21)[0], rmse(ds2grades, results22)[0])
#ap3rmse = (rmse(ds1grades, results31)[0], rmse(ds2grades, results32)[0])
#ap4rmse = (rmse(ds1grades, results41)[0], rmse(ds2grades, results42)[0])
#ap5rmse = (rmse(ds1grades, results51)[0], rmse(ds2grades, results52)[0])

#rq1(ap1rmse, ap2rmse, ap3rmse, ap4rmse, ap5rmse)
#rq3(dataset1, collectionlist1, ds1grades, itemcounts1, dataset1t, itemcounts1t, collectionlist1t, dataset2, collectionlist2, ds2grades, itemcounts2, dataset2t, itemcounts2t, collectionlist2t)



