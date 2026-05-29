#!unzip Collection\ 1\ -\ Temporary.zip
# !pip install gensim

import os
import random
import sys
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

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def parse_list(text):
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(p).strip() for p in parsed if str(p).strip()]
    except Exception as e:
        pass
    counter = 0
    if text.count('[') == 0 and text.count(']') == 1:
        counter = 1
    if text.count('[') == text.count(']') == 0:
        counter = 1
    # if text == "":
    #       counter = 2
    cleaned = text.replace('\n', ' ').replace('\r', ' ')
    cleaned = cleaned.strip().strip("[]")
    cleaned = re.sub(r'[:\-]+', ' ', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    parts = re.split(r"[,'\"]+", cleaned)
    parts = [p for p in parts if p]
    if len(parts) == 0:
        counter = 2
    if counter == 1:
        parts.append("APPENDME")
    if counter == 2:
        parts.append("DELETEME")
    return parts


def clean_dataset(path):
    ds1 = os.listdir(path)
    ds1files = []
    filenames = []
    for i in ds1:
        fi, ex = os.path.splitext(i)
        if ex.lower() == '.txt':
            ds1files.append(i)
            filenames.append(fi)
    dataset1 = []
    itemcounts = {}
    for it, i in enumerate(ds1files):
        counter = 0
        f = open(path + "/" + i, "r", encoding="utf-8")
        doc = []
        for line in f:
            counter += 1
            newline = parse_list(line)
            newline = [s.strip() for s in newline if s.strip()]
            # print(newline)
            # print(counter)
            if newline[len(newline) - 1] == "DELETEME" or newline[len(newline) - 1] == "APPENDME":
                counter -= 1
            if newline[len(newline) - 1] == "APPENDME" and counter != 0:
                # print("enter")
                doc[counter - 1].extend(newline[0:len(newline) - 1])
            elif newline[len(newline) - 1] != "DELETEME":
                doc.append(newline)

        # print(i)
        # print(doc)

        # ADD PREFIX OR NOT
        prefixed_doc = []

        itemcounts[filenames[it]] = {}
        for d in doc:
            if d[0] not in itemcounts[filenames[it]]:
                itemcounts[filenames[it]][d[0]] = 1
            else:
                itemcounts[filenames[it]][d[0]] += 1
            prefixed_line = []
            y = d[0] + '_'
            for x in d:
                prefixed_line.append(y + x)
            prefixed_doc.append(prefixed_line)
        dataset1.append((filenames[it], prefixed_doc))

        # USE THIS LINE IF NOT PREFIXING
        # dataset1.append((filenames[it], doc))
        f.close()
    listofitemtypes = []
    for item in itemcounts:
        for it in itemcounts[item]:
            if it not in listofitemtypes:
                listofitemtypes.append(it)
    for item in itemcounts:
        for it in listofitemtypes:
            if it not in itemcounts[item]:
                itemcounts[item][it] = 0
    return dataset1, itemcounts


def stemming(data):
    ps = PorterStemmer()

    allwordsfp = []
    for doc in data:
        allwords = []
        allwordsf = []
        for x in doc[1]:
            # print(x)
            if type(x) == str:
                allwords.append(x)
            elif type(x) == list:
                flow = ""
                for y in x:
                    flow += y + " "
                # print(flow)
                words1 = word_tokenize(flow)
                allwords.extend(words1)

        # print(allwords)
        for w in allwords:
            allwordsf.append(ps.stem(w))
        allwordsfp.append((doc[0], allwordsf))
        # print(allwordsf)
    return allwordsfp


def removestopwords(data):
    stop_words = set(stopwords.words('english'))

    allwordsfp = []
    for doc in data:
        allwords = []
        allwordsf = []
        for x in doc[1]:
            flow = ""
            for y in x:
                flow += y + " "
            # print(flow)
            # print(x)
            words = word_tokenize(flow.lower())
            # print(words)
            allwords.extend(words)
        # print(allwords)
        for w in allwords:
            if w not in stop_words:
                allwordsf.append(w)
        allwordsfp.append((doc[0], allwordsf))
        # print(allwordsf)
    return allwordsfp


def splittestingandtraining(datagrades, dataset, itemcount, testing=False):
    nums = []
    trainingdata1 = []
    testingdata1 = []
    trainingdata1names = []
    trainingdata1conts = []
    testingdata1names = []
    testingdata1conts = []
    trainitemcount = {}
    testitemcount = {}
    if testing is True:
        random.seed(87)
        nums = random.sample(range(1, len(datagrades) + 1), 70)
        if 89 in nums:
            nums.remove(89)
        if 62 in nums:
            nums.remove(62)
        # print(nums)
        for i in nums:
            if i == 89 or i == 62:
                print("KILL YOURSELF")
        for d in datagrades:
            if int(d[0]) in nums:
                trainingdata1.append(d)
        for d in datagrades:
            if d not in trainingdata1:
                testingdata1.append(d)
        for d in trainingdata1:
            for x in dataset:
                if d[0] == x[0]:
                    trainingdata1names.append(d[0])
                    trainingdata1conts.append(x[1])

        for d in testingdata1:
            for x in dataset:
                if d[0] == x[0]:
                    testingdata1names.append(d[0])
                    testingdata1conts.append(x[1])

        for item in itemcount:
            if int(item) in nums:
                trainitemcount[item] = itemcount[item]
        # print(trainitemcount)
        for item in itemcount:
            if item not in trainitemcount.keys():
                testitemcount[item] = itemcount[item]
    else:

        for d in datagrades:
            trainingdata1.append(d)

        for d in dataset:
            ticker = 0
            for x in trainingdata1:
                if d[0] == x[0]:
                    ticker = 1
            if ticker == 0:
                testingdata1.append((d[0], 0.0))
        for d in trainingdata1:
            for x in dataset:
                if d[0] == x[0]:
                    trainingdata1names.append(d[0])
                    trainingdata1conts.append(x[1])

        for d in testingdata1:
            for x in dataset:
                if d[0] == x[0]:
                    testingdata1names.append(d[0])
                    testingdata1conts.append(x[1])
        for item in itemcount:
            # print(item)
            # print(nums)
            for d in trainingdata1:
                if item == d[0]:
                    trainitemcount[item] = itemcount[item]
        # print(trainitemcount)
        for item in itemcount:
            if item not in trainitemcount.keys():
                testitemcount[item] = itemcount[item]

    return trainingdata1, testingdata1, trainingdata1names, trainingdata1conts, testingdata1names, testingdata1conts, trainitemcount, testitemcount


def countdistance(count1, count2):
    # print(count1)
    # print(count2)
    sum = 0
    counter = 0
    for item in count1:
        sum += abs(count1[item] - count2[item])
        counter += 1
    total = sum / counter
    distance = total
    return distance


def approach1(traindata, testdata, trainconts, testconts, trainnames, testnames, trainitemcount, testitemcount, k=3):
    # print(testnames)
    d1dictionary = corpora.Dictionary(trainconts + testconts)

    BoW_corpus_d1training = [d1dictionary.doc2bow(doc, allow_update=True) for doc in trainconts]
    BoW_corpus_d1testing = [d1dictionary.doc2bow(doc, allow_update=True) for doc in testconts]

    tfidf = gensim.models.TfidfModel(BoW_corpus_d1training + BoW_corpus_d1testing, smartirs='ltc')

    d1training_tfidf = [tfidf[doc] for doc in BoW_corpus_d1training]
    d1testing_tfidf = [tfidf[doc] for doc in BoW_corpus_d1testing]

    scores = gensim.similarities.MatrixSimilarity(d1training_tfidf, num_features=len(d1dictionary))

    finalresults = []
    for i, name in enumerate(testnames):
        counts = testitemcount[name]
        # print(counts)
        # for item in counts:
        #   print(item, testitemcount[name][item])
        traindistances = {}
        for item in trainitemcount:
            # print(item)
            counttr = trainitemcount[item]
            dis = countdistance(counts, counttr)
            if dis != 0:
                traindistances[item] = dis
            else:
                traindistances[item] = 0
            # print(dis)
            # print(counttr)
            # dis = countdistance(counts, counttr)

        sim = scores[d1testing_tfidf[i]]
        namescore = []
        for h in range(len(sim)):
            namescore.append((trainnames[h], sim[h]))
        fullnamescore = []
        for name1 in namescore.copy():
            fullnamescore.append((name1[0], name1[1], traindistances[name1[0]]))
        avgnamescore = []
        for full in fullnamescore:
            avger = (full[1] + full[2]) / 2
            avgnamescore.append((full[0], avger))

        sortednamescore = sorted(avgnamescore, key=lambda x: x[1], reverse=True)
        # IF NOT USING COUNTS UNCOMMENT THIS LINE
        # sortednamescore = sorted(namescore, key=lambda x: x[1], reverse=True)
        # print(sortednamescore)

        # print(sortednamescore)
        knnlist = sortednamescore[0:k]
        # print(knnlist)
        sum = 0
        for x in knnlist:
            for d in traindata:
                if x[0] == d[0]:
                    sum += d[1]
        score = sum / k
        # print((name, score))
        finalresults.append((name, score))
    return finalresults


def approach2(traindata, testdata, trainconts, testconts, trainnames, testnames, trainitemcount, testitemcount, k=3):
    longtrain = []
    longtest = []
    for d in range(len(trainconts)):
        longt = ""
        for x in trainconts[d]:
            longt += x
        longtrain.append((trainnames[d], longt))
    for d in range(len(testconts)):
        longt = ""
        for x in testconts[d]:
            longt += x
        longtest.append((testnames[d], longt))

    traindocuments = [TaggedDocument(doc, [i]) for i, doc in longtrain]
    testdocuments = [TaggedDocument(doc, [i]) for i, doc in longtest]

    model = Doc2Vec(traindocuments + testdocuments, vector_size=5, window=2, min_count=1, workers=4)
    # for name in testnames:
    #      print(model.dv.most_similar(name))
    # scores = gensim.similarities.MatrixSimilarity(traindocuments, num_features=5)

    finalresults = []
    for i, name in enumerate(testnames):
        counts = testitemcount[name]
        # print(counts)
        # for item in counts:
        #   print(item, testitemcount[name][item])
        traindistances = {}
        for item in trainitemcount:
            # print(item)
            counttr = trainitemcount[item]
            dis = countdistance(counts, counttr)
            traindistances[item] = dis
            # print(dis)
            # print(counttr)
            # dis = countdistance(counts, counttr)

        # sim = scores[testdocuments[i]]
        sim = model.dv.most_similar(name)
        namescore = []
        for h in range(len(sim)):
            if sim[h][0] in trainnames:
                namescore.append(sim[h])

        fullnamescore = []
        for name1 in namescore.copy():
            fullnamescore.append((name1[0], name1[1], traindistances[name1[0]]))
        avgnamescore = []
        for full in fullnamescore:
            avger = (full[1] + full[2]) / 2
            avgnamescore.append((full[0], avger))

        sortednamescore = sorted(avgnamescore, key=lambda x: x[1], reverse=True)
        # print(namescore)
        # sortednamescore = sorted(namescore, key=lambda x: x[1], reverse=True)
        # print(sortednamescore)
        knnlist = sortednamescore[0:k]
        # print(knnlist)
        sum = 0
        for x in knnlist:
            for d in traindata:
                if x[0] == d[0]:
                    sum += d[1]
        score = sum / k
        # print((name, score))
        finalresults.append((name, score))
    return finalresults


def finderror(grades, results):
    combos = []
    for r in results:
        for g in grades:
            if r[0] == g[0]:
                combos.append((r[0], g[1], r[1]))
    # print(grades)
    # print(results)
    se = 0
    for c in combos:
        se += (c[2] - c[1]) ** 2
    rmse = math.sqrt(se / len(combos))
    ae = 0
    for c in combos:
        ae += abs(c[2] - c[1])
    mae = ae / len(combos)
    return rmse, mae


# ADD NEW PATH TO HERE TO CHANGE FILE. IF ONLY ONE DATASET, ONLY USE ds1path
#ds1path = "drive/MyDrive/Colab Notebooks/Collection1Temp/Dataset1"
#ds2path = "drive/MyDrive/Colab Notebooks/Collection1Temp/Dataset2"
ds1path = "Collection1Temp/Dataset1"
ds2path = "Collection1Temp/Dataset2"
dataset1, itemcounts1 = clean_dataset(ds1path)
dataset2, itemcounts2 = clean_dataset(ds2path)
# ADD NEW PATH HERE
f = open("Collection1Temp/ERD_grades.csv")
ds1grades = []
ds2grades = []
count = 0
for line in f:
    parts = line.split(",")
    if count != 0:
        ds1grades.append((parts[0], float(parts[1])))
        ds2grades.append((parts[0], float(parts[2])))
    count += 1

# REMOVAL OF STOP WORDS (separates all words into one list of individual words)
print("Removing stop words")
dataset1 = removestopwords(dataset1)
dataset2 = removestopwords(dataset2)

# STEMMING (separates all words into one list of individual words)
print("stemming words")
dataset1 = stemming(dataset1)
dataset2 = stemming(dataset2)

# TESTING AND TRAINING NOT WITH SUBMISSION
testing = False
trainingdata1, testingdata1, trainingdata1names, trainingdata1conts, testingdata1names, testingdata1conts, trainitemcount1, testitemcount1 = splittestingandtraining(
    ds1grades, dataset1, itemcounts1, testing=testing)
trainingdata2, testingdata2, trainingdata2names, trainingdata2conts, testingdata2names, testingdata2conts, trainitemcount2, testitemcount2 = splittestingandtraining(
    ds2grades, dataset2, itemcounts2, testing=testing)
testitemcount1 = dict(sorted(testitemcount1.items()))
trainitemcount1 = dict(sorted(trainitemcount1.items()))
testitemcount2 = dict(sorted(testitemcount2.items()))
trainitemcount2 = dict(sorted(trainitemcount2.items()))

# ESTABLISH K
k = 4

d1results = approach1(trainingdata1, testingdata1, trainingdata1conts, testingdata1conts, trainingdata1names,
                      testingdata1names, trainitemcount1, testitemcount1, k)
d2results = approach1(trainingdata2, testingdata2, trainingdata2conts, testingdata2conts, trainingdata2names,
                      testingdata2names, trainitemcount2, testitemcount2, k)

print(d1results)
print(d2results)
# IF TESTING RMSE AND MAE WILL BE CALCULATED AND DISPLAYED
if testing:
    rmse1, mae1 = finderror(ds1grades, d1results)
    rmse2, mae2 = finderror(ds2grades, d2results)

    print("RMSE for dataset 1: " + str(rmse1) + " MAE for dataset 1: " + str(mae1))

    print("RMSE for dataset 2: " + str(rmse2) + " MAE for dataset 2: " + str(mae2))

d1results2 = approach2(trainingdata1, testingdata1, trainingdata1conts, testingdata1conts, trainingdata1names,
                       testingdata1names, trainitemcount1, testitemcount1, k)
d2results2 = approach2(trainingdata2, testingdata2, trainingdata2conts, testingdata2conts, trainingdata2names,
                       testingdata2names, trainitemcount2, testitemcount2, k)

print(d1results2)
print(d2results2)
# IF TESTING RMSE AND MAE WILL BE CALCULATED AND DISPLAYED
if testing:
    rmse12, mae12 = finderror(ds1grades, d1results2)
    rmse22, mae22 = finderror(ds2grades, d2results2)

    print("RMSE for dataset 1: " + str(rmse12) + " MAE for dataset 1: " + str(mae12))

    print("RMSE for dataset 2: " + str(rmse22) + " MAE for dataset 2: " + str(mae22))


def sortforcopying(results):
    re = sorted(results, key=lambda x: x[0], reverse=False)
    return re


# SET one AND two TO VARIABLE NAME FOR RESULTS FROM BOTH DATASETS. CHANGE ENDING OF VARIABLE FOR OTHER APPROACH
one = sortforcopying(d1results)
two = sortforcopying(d2results)

for w in range(len(one)):
    print(one[w][0] + "," + str(one[w][1]) + "," + str(two[w][1]))
# USE THIS INSTEAD IF ONLY ONE DATASET TO GRADE
# for w in range(len(one)):
#   print(one[w][0]+ "," + str(one[w][1]))









