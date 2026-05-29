import os
import random
import sys
from dataclasses import replace
from time import process_time_ns

import nltk
import gensim
import numpy as np
import csv
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from numpy import array, dot
from numpy.linalg import norm
import ast
import re
import math
import json
import difflib
#from utilities import removestopwords, stemming, normalize_data, rmse, clean_dataset
from gensim.models.word2vec import Word2Vec

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def entity_similarity(e1, e2, model, natrs1, natrs2, params):
    ksc = params['entity_type_c']
    A = params['A']
    B = params['B']
    C = params['C']
    D = params['D']

    #print(e1['kind'], e2['kind'])
    #ks = model.wv.similarity(e1['kind'], e2['kind'])
    if e1['kind'] == e2['kind']:
        ks = 1
    else:
        ks = ksc
    ns = model.wv.similarity(e1['name'], e2['name'])
    if natrs1 == 0 and natrs2 == 0:
        atrs = 1
    else:
        atrs = abs(natrs1-natrs2) / max(natrs1, natrs2)

    e1atrembs = []
    for a in e1['attributes']:
        e1atrembs.append(model.wv[a])
    e2atrembs = []
    for a in e2['attributes']:
        e2atrembs.append(model.wv[a])
    if len(e1atrembs) == 0 or len(e2atrembs) == 0:
        atrembs = 0
    else:
        e1atrembs = array(e1atrembs)
        e2atrembs = array(e2atrembs)
        avge1atrembs = sum(e1atrembs)/len(e1atrembs)
        avge2atrembs = sum(e2atrembs)/len(e2atrembs)

        atrembs = dot(avge1atrembs, avge2atrembs) / (norm(avge1atrembs) * norm(avge2atrembs))


    #print(ks, ns, atrs, atrembs)


    es = A * ks + B * ns + C * atrs + D * atrembs
    return es



# returns a dictionary lists of entity similarity for each test doc. format is dict[testdoc['erd_id']] = [(traindoc['erd_id], similarity), ...]
def run_entity_similarity(doc, doc2, model, params):
    entity_pairs = []
    pickedalreadyentities = []

    entity_pair_similarity = []
    for e1 in doc['entities']:
        for e2 in doc2['entities']:
            # print(e1['name'], e2['name'])
            sim = entity_similarity(e1, e2, model, len(e1['attributes']), len(e2['attributes']), params)

            entity_pair_similarity.append((e1, e2, sim))
        entity_pair_similarity = sorted(entity_pair_similarity, key=lambda x: x[2], reverse=True)
    for x in entity_pair_similarity:
        if x[1] not in pickedalreadyentities:
            entity_pairs.append((x[0], x[1], x[2]))
            pickedalreadyentities.append(x[1])

    avg = 0
    for x in entity_pairs:
        if x[1] != None:
            avg += x[2]
    similarity = avg / len(entity_pairs)

    return similarity







def relationship_similarity(rel1, rel2, model, rel1entities, rel2entities, params):
    T = params["T"]
    U = params["U"]
    V = params["V"]
    X = params["X"]
    Y = params["Y"]
    Z = params["Z"]
    tsc = params["rel_type_c"]
    #type similarity ks
    if rel1['kind'] == rel2['kind']:
        ks = 1.0
    else:
        ks = tsc

    #arity similarity ars
    nrie1 = len(rel1['involved_entities'])
    nrie2 = len(rel2['involved_entities'])
    ars = (abs(nrie1 - nrie2) / ((nrie1 + nrie2)))

    #similarity of participating entities ies
    tags = ['weak_entity_', 'entity_', 'relationship_', 'identifying_relationship_']
    r1ief = []
    for x in rel1['involved_entities']:
        for ent in rel1entities:
            entname = ent['name']
            for t in tags:
                if t in ent['name']:
                    parts = ent['name'].split(t,1)
                    entname = parts[1]
            #print(entname, x['name'])
            if entname == x['name']:
                r1ief.append(ent)
    r2ief = []
    for x in rel2['involved_entities']:
        for ent in rel2entities:
            entname = ent['name']
            for t in tags:
                if t in ent['name']:
                    parts = ent['name'].split(t,1)
                    entname = parts[1]
            #print(ent['name'], x['name'])
            if entname == x['name']:
                r2ief.append(ent)

    entity_pairs = []
    pickedalreadyentities = []
    entity_pair_similarity = []
    #print(r1ief)
    #print(r2ief)
    for e1 in r1ief:
        for e2 in r2ief:
            # print(e1['name'], e2['name'])
            sim = entity_similarity(e1, e2, model, len(e1['attributes']), len(e2['attributes']), params)

            entity_pair_similarity.append((e1, e2, sim))
        entity_pair_similarity = sorted(entity_pair_similarity, key=lambda x: x[2], reverse=True)
    for x in entity_pair_similarity:
        if x[1] not in pickedalreadyentities:
            entity_pairs.append((x[0], x[1], x[2]))
            pickedalreadyentities.append(x[1])

    avg = 0
    #print(entity_pairs)
    if len(entity_pairs) > 0:
        for x in entity_pairs:
            if x[1] != None:
                avg += x[2]
        ies = avg / len(entity_pairs)
    else:
        ies = 0

    #similarity of attributes atrembs
    atr1embs = []
    for atr1 in rel1['attributes']:
        atr1embs.append(model.wv[atr1])
    atr2embs = []
    for atr2 in rel2['attributes']:
        atr2embs.append(model.wv[atr2])
    if len(atr1embs) == 0 or len(atr2embs) == 0:
        atrembs = 0
    else:
        atr1embs = array(atr1embs)
        atr2embs = array(atr2embs)
        atr1s = sum(atr1embs)/len(atr1embs)
        atr2s = sum(atr2embs)/len(atr2embs)

        atrembs = dot(atr1s, atr2s) / (norm(atr1s) * norm(atr2s))


    possible_cardinalities = ['01', '11', '1M', '0M', 'UNC']

    #similarity of max cardinality
    #similarity of min cardinality
    max_cardinality1 = ""
    max_cardinality2 = ""
    min_cardinality1 = ""
    min_cardinality2 = ""
    if len(rel1['involved_entities']) == 2:
        if rel1['involved_entities'][0]['cardinality'] != 'UNC' and rel1['involved_entities'][0]['cardinality'] in possible_cardinalities and rel1['involved_entities'][1]['cardinality'] in possible_cardinalities:
            max_cardinality1 = rel1['involved_entities'][0]['cardinality'][1] + \
                                rel1['involved_entities'][1]['cardinality'][1]
            min_cardinality1 = rel1['involved_entities'][0]['cardinality'][0] + \
                                rel1['involved_entities'][1]['cardinality'][0]
    elif len(rel1['involved_entities']) > 2:
        max_cardinality1 = 'Bigger'
        min_cardinality1 = 'Bigger'
    else:
        max_cardinality1 = 'Unknown'
        min_cardinality1 = 'Unknown'

    if len(rel2['involved_entities']) == 2:
        if rel2['involved_entities'][0]['cardinality'] != 'UNC' and rel2['involved_entities'][0]['cardinality'] in possible_cardinalities and rel2['involved_entities'][1]['cardinality'] in possible_cardinalities:
            max_cardinality2 = rel2['involved_entities'][0]['cardinality'][1] + \
                               rel2['involved_entities'][1]['cardinality'][1]
            min_cardinality2 = rel2['involved_entities'][0]['cardinality'][0] + \
                               rel2['involved_entities'][1]['cardinality'][0]
    elif len(rel2['involved_entities']) > 2:
        max_cardinality2 = 'Bigger'
        min_cardinality2 = 'Bigger'
    else:
        max_cardinality2 = 'Unknown'
        min_cardinality2 = 'Unknown'

    if len(max_cardinality1) == 2 and len(max_cardinality2) == 2:
        difcount = 0
        for i,c in enumerate(max_cardinality1):
            if max_cardinality1[i] != max_cardinality2[i]:
                difcount += 1
        maxcdis = difcount / len(max_cardinality1)
    elif max_cardinality1 == 'Bigger' and max_cardinality2 == 'Bigger':
        maxcdis = 0
    else:
        maxcdis = 1

    maxcs = 1-maxcdis
    if len(min_cardinality1) == 2 and len(min_cardinality2) == 2:
        difcount = 0
        for i,c in enumerate(min_cardinality1):
            if min_cardinality1[i] != min_cardinality2[i]:
                difcount += 1
        mincdis = difcount / len(min_cardinality1)
    elif max_cardinality1 == 'Bigger' and max_cardinality2 == 'Bigger':
        mincdis = 0
    else:
        mincdis = 1

    mincs = 1-mincdis




    similarity = T*ks + U*ars + V*ies + X*atrembs + Y*maxcs + Z*mincs

    return similarity

def run_relationship_similarity(doc1, doc2, model, params):

    rel_pairs = []
    pickedalreadyrels = []

    rel_pair_similarity = []
    for rel1 in doc1['relationships']:
        for rel2 in doc2['relationships']:
            # print(e1['name'], e2['name'])
            rel1entities = doc1['entities']
            rel2entities = doc2['entities']
            sim = relationship_similarity(rel1, rel2, model, rel1entities, rel2entities, params)

            rel_pair_similarity.append((rel1, rel2, sim))
        rel_pair_similarity = sorted(rel_pair_similarity, key=lambda x: x[2], reverse=True)
    for x in rel_pair_similarity:
        if x[1] not in pickedalreadyrels:
            rel_pairs.append((x[0], x[1], x[2]))
            pickedalreadyrels.append(x[1])

    avg = 0
    for x in rel_pairs:
        if rel_pairs[1] != None:
            avg += x[2]
    similarity = avg / len(rel_pairs)

    return similarity

def additional_features(dataset):
    itemcounts = {}
    for doc in dataset:
        itemcounts[doc['erd_id']] = {}
        for e in doc['entities']:
            if e['kind'] not in itemcounts.keys():
                if e['kind'] == 'weak_entity':
                    itemcounts[doc['erd_id']][e['kind']] = 1
                else:
                    itemcounts[doc['erd_id']][e['kind']] = 1
            else:
                itemcounts[doc['erd_id']][e['kind']] += 1
        itemcounts[doc['erd_id']]["relationship_attributes"] = 0
        itemcounts[doc['erd_id']]["binary_entity_entity"] = 0
        itemcounts[doc['erd_id']]["n_way_entity_entity"] = 0
        itemcounts[doc['erd_id']]["binary_weak_entity_weak_entity"] = 0
        itemcounts[doc['erd_id']]["n_way_weak_entity_weak_entity"] = 0
        itemcounts[doc['erd_id']]["binary_relationship_relationship"] = 0
        itemcounts[doc['erd_id']]["n_way_relationship_relationship"] = 0
        itemcounts[doc['erd_id']]["binary_identifying_relationship_identifying_relationship"] = 0
        itemcounts[doc['erd_id']]["n_way_identifying_relationship_identifying_relationship"] = 0

        for d in doc['relationships']:
            kind = d['kind']
            n = len(d['involved_entities'])
            itemcounts[doc['erd_id']]["relationship_attributes"] += len(d['attributes'])
            if kind not in itemcounts[doc['erd_id']].keys():
                if kind == 'identifying_relationship':
                    itemcounts[doc['erd_id']][kind] = 1
                else:
                    itemcounts[doc['erd_id']][kind] = 1
            else:
                itemcounts[doc['erd_id']][kind] += 1

            if n == 2 and n is not None:
                itemcounts[doc['erd_id']][f"binary_{kind}"] += 1
            elif n > 2 and n is not None:
                itemcounts[doc['erd_id']][f"n_way_{kind}"] += 1
    for item in itemcounts.keys():
        if 'entity_entity' not in itemcounts[item].keys():
            itemcounts[item]["entity_entity"] = 0
        if 'weak_entity_weak_entity' not in itemcounts[item].keys():
            itemcounts[item]["weak_entity_weak_entity"] = 0
        if 'relationship_relationship' not in itemcounts[item].keys():
            itemcounts[item]["relationship_relationship"] = 0
        if 'identifying_relationship_identifying_relationship' not in itemcounts[item].keys():
            itemcounts[item]["identifying_relationship_identifying_relationship"] = 0

    #now have item counts for each doc

    return itemcounts

def cos_sim(erd1, erd2):
    dot_prod = 0
    for key in erd1.keys():
        dot_prod += erd1[key] * erd2[key]
    mag1 = np.linalg.norm(list(erd1.values()))
    mag2 = np.linalg.norm(list(erd2.values()))

    return dot_prod/ (mag1 * mag2)



def approach4(dataset, collection, testdata, traindata, testconts, trainconts, trainnames, testnames, trainitems, testitems, grades, params, k=3, threshold=.80):
    #print(testconts)
    #print(trainconts)

    alpha = params["alpha"]
    beta = params["beta"]
    lamb = params["lambda"]
    #k = params["K"]
    model = Word2Vec(testconts + trainconts, vector_size=100, window=5, min_count=1, workers=4)

    entity_similarity_per_doc = {}
    final_results = []
    for doc in testdata:
        similar_docs = []
        for doc2 in traindata:
            if doc == doc2:
                continue
            #print(doc['erd_id'], doc2['erd_id'])
            entity_similarity = run_entity_similarity(doc, doc2, model, params)
            relation_similarity = run_relationship_similarity(doc, doc2, model, params)

            itemcounts1 = additional_features(testdata)
            itemcounts2 = additional_features(traindata)
            itemcount1 = itemcounts1[doc['erd_id']]
            itemcount2 = itemcounts2[doc2['erd_id']]
            additional_feature_similarity = cos_sim(itemcount1, itemcount2)

            #print(entity_similarity, relation_similarity, additional_feature_similarity)

            similarity = alpha * entity_similarity + beta * relation_similarity + lamb * additional_feature_similarity
            nameparts = doc2['erd_id'].split('_')

            similar_docs.append((nameparts[0],similarity))
        #print(similar_docs[:k])
        sortednamescore = sorted(similar_docs, key=lambda x: x[1], reverse=True)
        #print(k)
        #print(sortednamescore)
        knnlist = sortednamescore[:k]

        for i,ki in enumerate(knnlist):
            if ki[1] < threshold:
                knnlist.remove(ki)

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
            score = sumt / sumb
        testname = doc['erd_id'].split('_')[0]
        final_results.append((testname, score))

    return final_results

#modified version of approach 4 for the sake of research question 3
def approach4_1(dataset, collection, testdata, traindata, testconts, trainconts, trainnames, testnames, trainitems, testitems, grades, params, k=3, threshold=.80):
    #print(testconts)
    #print(trainconts)

    alpha = params["alpha"]
    beta = params["beta"]
    lamb = params["lambda"]
    #k = params["K"]
    model = Word2Vec(testconts + trainconts, vector_size=100, window=5, min_count=1, workers=4)

    entity_similarity_per_doc = {}
    final_results = []
    for doc in testdata:
        similar_docs = []
        for doc2 in traindata:
            if doc == doc2:
                continue
            #print(doc['erd_id'], doc2['erd_id'])
            entity_similarity = run_entity_similarity(doc, doc2, model, params)
            relation_similarity = run_relationship_similarity(doc, doc2, model, params)

            itemcounts1 = additional_features(testdata)
            itemcounts2 = additional_features(traindata)
            itemcount1 = itemcounts1[doc['erd_id']]
            itemcount2 = itemcounts2[doc2['erd_id']]
            additional_feature_similarity = cos_sim(itemcount1, itemcount2)

            #print(entity_similarity, relation_similarity, additional_feature_similarity)

            similarity = alpha * entity_similarity + beta * relation_similarity + lamb * additional_feature_similarity
            nameparts = doc['erd_id'].split('_')
            nameparts2 = doc2['erd_id'].split('_')
            if nameparts[0] != nameparts2[0]:
                final_results.append((nameparts[0],nameparts2[0], similarity))
        #print(similar_docs[:k])
    sortednamescore = sorted(final_results, key=lambda x: x[2], reverse=True)
    top20 = sortednamescore[:20]


    return top20