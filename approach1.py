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
from utilities import removestopwords, stemming, normalize_data, rmse, clean_dataset, splits
from approach4 import run_entity_similarity, run_relationship_similarity, approach4

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer







#calculate distances between different item counts. very much open to different calculations
def countdistance(count1, count2):
    #print(count1)
    #print(count2)
    sum = 0
    counter = 0

    for item in count1:
        sum += abs(count1[item] - count2[item])
        counter += 1
    total = sum / counter
    distance = total
    #print(distance)
    return distance

def count_similarity_weighted(count1, count2, max_ranges, importance, eps=1e-9):
    keys = set(count1.keys()) | set(count2.keys())
    if not keys:
        return 1.0

    weighted_diffs = 0.0
    total_weight = 0.0

    for k in keys:
        v1 = float(count1.get(k,0))
        v2 = float(count2.get(k,0))
        diff = abs(v1 - v2)

        if max_ranges and k in max_ranges and max_ranges[k] > 0:
            denom = max_ranges[k]
        else:
            denom = max(v1, v2, 1.0)

        norm_diff = diff/(denom+eps)

        w = 1.0
        if importance and k in importance:
            w = float(importance[k])

        weighted_diffs += norm_diff*w
        total_weight += w
    if total_weight == 0:
        avg_norm = 0.0
    else:
        avg_norm = weighted_diffs/total_weight

    similarity = 1.0 - avg_norm

    if similarity < 0: similarity = 0.0
    if similarity > 1: similarity = 1.0
    return similarity


def compute_max_ranges(itemcounts):
    max_ranges = {}
    for fname, counts in itemcounts.items():
        for k, v in counts.items():
            max_ranges[k] = max(max_ranges.get(k, 0), v)
    for k in list(max_ranges.keys()):
        if max_ranges[k] == 0:
            max_ranges[k] = 1.0
    return max_ranges

def tfidf_to_dense(tfidf_vec, vocab_size):
    dense = np.zeros(vocab_size)
    for idx, weight in tfidf_vec:
        dense[idx] = weight
    return dense
#approach 1
def approach1(traindata, testdata, trainconts, testconts, trainnames, testnames, trainitemcounts, testitemcounts, grades, params, k=3):
    #dictionary = corpora.Dictionary(testconts + trainconts)
    dictionary = corpora.Dictionary(trainconts)
    testcorpus = [dictionary.doc2bow(doc, allow_update=True) for doc in testconts]
    traincorpus = [dictionary.doc2bow(doc, allow_update=True) for doc in trainconts]
    #print(trainconts)
    #tfidf = gensim.models.TfidfModel(testcorpus + traincorpus, smartirs='ltc')
    tfidf = gensim.models.TfidfModel(traincorpus, smartirs='ltc')
    #print(tfidf)
    train_tfidf = [tfidf[doc] for doc in traincorpus]
    test_tfidf = [tfidf[doc] for doc in testcorpus]

    train_dense = np.array([tfidf_to_dense(tfidf[doc], len(dictionary)) for doc in traincorpus])
    test_dense = np.array([tfidf_to_dense(tfidf[doc], len(dictionary)) for doc in testcorpus])
    #print(train_tfidf)
    pca = PCA(n_components=50, random_state=42)
    train_pca = pca.fit_transform(train_dense)
    test_pca = pca.transform(test_dense)
    #scores = gensim.similarities.MatrixSimilarity(train_tfidf, num_features= len(dictionary))
    #scores = gensim.similarities.MatrixSimilarity(tfidf_pca, num_features=len(dictionary))
    similarity_matrix = cosine_similarity(test_pca, train_pca)
    #print(scores)
    finalresults = []

    max_ranges = compute_max_ranges(trainitemcounts)
    importance = {'entity': params['A'], 'weak_entity': params['B'], 'relationship': params['C'], 'identifying_relationship': params['D']}
    alpha = params['alpha']
    beta = params['beta']
    #importance = {'entity': 2.0, 'weak_entity': 1.5, 'relationship': 2.25, 'identifying_relationship': 1.2}
    #alpha = .45
    for i, name in enumerate(testnames):
        counts = testitemcounts[name]
        traindistances = {}
        for item in trainitemcounts:
            counttr = trainitemcounts[item]
            sim_counts = count_similarity_weighted(counts, counttr, max_ranges=max_ranges, importance=importance)
            # dis = countdistance(counts, counttr)
            # if dis != 0:
            #     traindistances[item] = dis
            # else:
            #     traindistances[item] = 0
            traindistances[item] = sim_counts



        #sim = scores[test_tfidf[i]]
        sim = similarity_matrix[i]
        #print(sim)
        namescore = []
        for h in range(len(sim)):
            namescore.append((trainnames[h], float(sim[h])))
        # fullnamescore = []
        # for name1 in namescore.copy():
        #     fullnamescore.append((name1[0], name1[1], traindistances[name1[0]]))
        # avgnamescore = []
        # for full in fullnamescore:
        #     avger = (full[1] + full[2]) / 2
        #     avgnamescore.append((full[0], avger))
        combined = []
        for tn,tfidf_sim in namescore:
            count_sim = traindistances.get(tn,0.0)
            combined_sim = alpha * tfidf_sim + (1 - alpha) * count_sim
            combined.append((tn,combined_sim))

        sortednamescore = sorted(combined, key=lambda x: x[1], reverse=True)
        #sortednamescore = sorted(avgnamescore, key=lambda x: x[1], reverse=True)

        knnlist = sortednamescore[:k]

        if 'comment' in testdata[i].keys():
            ncomments = len(testdata[i]['comment']['flags'])
        else:
            ncomments = 0

        #print(knnlist)
        sumt = 0.0
        sumb = 0.0
        grade_map = dict(grades)
        for neighbor_name, weight in knnlist:
            if neighbor_name in grade_map:
                sumt += weight * grade_map[neighbor_name]
                sumb += weight
        if sumb == 0.0:
            score = np.mean([g for _, g in grades])
        else:
            score = (sumt/sumb) - (beta * ncomments)
        # for x in knnlist:
        #     for d in grades:
        #         if x[0] == d[0]:
        #             sumt += x[1] * d[1]
        #             sumb += x[1]
        # if sumb == 0:
        #     sumb = 1
        # score = sumt / sumb
        finalresults.append((name,score))

    """results are of format [(name, grade), ...]"""
    return finalresults

