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
import matplotlib.pyplot as plt
import ast
import re
import math
import json
import subprocess
import difflib
from approach4 import approach4, approach4_1
from approach3 import approach3
from approach5 import approach5
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer



#word replacement for better normalization
def replaceword(word):

    if word == 'weak entity':
        return 'weak_entity'
    if word == 'weak-entity':
        return 'weak_entity'
    if word == 'identifying relationship':
        return 'identifying_relationship'
    if word == 'elationship':
        return 'relationship'
    if '(' in word:
        word = word.replace('(', "")
    if ')' in word:
        word = word.replace(')', "")
        return word.lower()
    if word == "#":
        return "number"
    if word == "num":
        return "number"
    if len(word) > 1:
        if word[len(word)-1].isdigit():
            return word[:len(word)-1]
    if word != '' and '_' in word:
        if word[word.find("_")].isupper() and re.fullmatch(r'(?:[a-z_]+)?[A-Z][a-z]*(?:[A-Z][a-z]*)*', word):
            s1 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', word)
            word = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s1)
            return word.lower()
    if re.fullmatch(r'^(?:[a-z_]+)?[a-z]+([A-Z][a-z0-9]*)*$', word) is not None:
        s1 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', word)
        word = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s1)
        return word.lower()
    else:
        return word




#perform normalization on individual words
def normalize_word(word):
    #w = word.lower()
    w = [r.split('-') for r in word.split(" ")]
    flat = [item for sublist in w for item in sublist]
    #print(flat)

    if len(flat) > 1:
        word = ""
        for f in flat:
            f = replaceword(f)
            word+="_"+f
        word = word.lower().strip("_")

    else:
        word = replaceword(flat[0]).lower()

    #print(word)
    return word



#spelling fixes
def normalize_word2(word, freqs):
    n = max(1, int(len(freqs) * 0.60))
    frontn = len(freqs)-n
    rare = freqs[-n:]
    common = freqs[:frontn]
    commons = []

    for c, i in common:
        commons.append(c)
    for w,c in rare:
        if w == word:
            newword = word
            # newword = ""
            # matches = difflib.get_close_matches(word, commons, n=1, cutoff=.95)
            # if matches:
            #     newword = matches[0]
            # else:
            #     newword = word
            word = newword

    return word



#quick usage for easy access
#usage mainly for dataset
def datawordnormal(word, freqs):
    w = normalize_word(word)
    w = normalize_word2(w, freqs)
    #REMOVE HERE IF USING OR NOT (DATASET VERSION)
    #w = removestopwords(w)
    #w = stemming(w)
    return w


#perform word normalization on the whole dataset/collection
#returns dataset of form list of the json dictionaries ([{}, {}, ...])
#each dictionary inside has the keys of the associated file
#also return a collection of format described in build_collection
def normalize_data(dataset):
    collection = build_collection(dataset)
    collect = []
    for doc in collection:
        words = []
        for word in doc:
            word = normalize_word(word)
            words.append(word.lower())
        collect.append(words)
    #print(collection)

    dictionary = corpora.Dictionary(collect)
    corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in collect]
    #for i,c in dictionary.cfs.items():
    #    print(dictionary[i],c)

    wordfreqs = sorted([(dictionary[i],c) for i,c in dictionary.cfs.items()], key=lambda x:x[1], reverse=True)
    #print(wordfreqs)

    collect2 = []
    for doc in collect:
        words2 = []
        for word in doc:
            word = normalize_word2(word, wordfreqs)
            #REMOVE HERE IF USING OR NOT (COLLECTIO VERSION)
            #word2 = removestopwords(word)
            #word3 = stemming(word2)
            words2.append(word)
        collect2.append(words2)

    # xi = 0
    # for i, d in enumerate(dataset):
    #     if d['erd_id'] == '92_1.png':
    #         xi = i
    # print(collect[xi])

    newdataset = []
    for doc in dataset:
        newdoc = {}
        #for i in range(len(doc['entities'])):
        for types in doc.keys():
            if types == 'entities':
                newdoc['entities'] = []
                for e in doc['entities']:
                    newentity = {}
                    for k in e.keys():
                        if isinstance(e[k], list):
                            newentity[k] = []
                            for x in e[k]:
                                #print(x)
                                newentity[k].append(replaceword(e['kind'])+"_"+datawordnormal(x,wordfreqs))
                        else:
                            #print(e[k])
                            newentity[k] = replaceword(e['kind'])+"_"+datawordnormal(replaceword(e[k]), wordfreqs)
                            #docwords.append(e[k])
                    #print(newentity)
                    #print(newentity)
                    newdoc['entities'].append(newentity)
            elif types == 'relationships':
                newdoc['relationships'] = []
                for r in doc['relationships']:
                    newrel = {}
                    for k in r.keys():
                        if isinstance(r[k], list):
                            newrel[k] = []
                            for x in r[k]:
                                if isinstance(x, dict):
                                    newrel[k].append({"name": datawordnormal(x['name'],wordfreqs), "cardinality": x['cardinality']})
                                    #docwords.append(x['name'])
                                else:
                                    newrel[k].append(replaceword(r['kind'])+"_"+datawordnormal(x,wordfreqs))
                        else:
                            newrel[k] = replaceword(r['kind'])+"_"+datawordnormal(r[k], wordfreqs)
                            #docwords.append(r[k])
                    newdoc['relationships'].append(newrel)
            else:
                newdoc[types] = doc[types]

        #print(newdoc)
        newdataset.append(newdoc)

    #print(newdataset)


    return newdataset, collect2


#build collection of all words in the dataset into lists of words per file
#returns list of format [['word', 'word', ...], ['word', 'word', ...], ...]
def build_collection(dataset):
    collectionwords = []
    for doc in dataset:
        docwords = []
        #for i in range(len(doc['entities'])):
        for e in doc['entities']:
            for k in e.keys():
                if isinstance(e[k], list):
                    for x in e[k]:
                        #print(x)
                        docwords.append(replaceword(e['kind'])+"_"+normalize_word(x))
                else:
                    #print(e[k])
                    docwords.append(replaceword(e['kind'])+"_"+normalize_word(e[k]))
        for r in doc['relationships']:
            for k in r.keys():
                if isinstance(r[k], list):
                    for x in r[k]:
                        if isinstance(x, dict):
                            docwords.append(replaceword(r['kind'])+"_"+normalize_word(x['name']))
                        else:
                            docwords.append(replaceword(r['kind'])+"_"+normalize_word(x))
                else:
                    docwords.append(replaceword(r['kind'])+"_"+normalize_word(r[k]))
        collectionwords.append(docwords)

    return collectionwords

#take a file path and turn it into a dataset of json objects
def clean_dataset(path = "C:/Users/learj/PycharmProjects/pythonProject/cs473/project/for_students/for_students/Dataset1"):
    ds1 = os.listdir(path)
    #print(ds1)
    ds1files = []
    filenames = []
    for i in ds1:
        fi, ex = os.path.splitext(i)
        if ex.lower() == '.json':
            ds1files.append(i)
            filenames.append(fi)
    dataset1 = []
    itemcounts = {}
    #print(ds1files)
    for it, i in enumerate(ds1files):
        f = open(path + "/" + i, "r")
        data = json.load(f)
        dataset1.append(data)

        itemcounts[filenames[it]] = {}
        for d in data['entities']:
            if d['kind'] not in itemcounts[filenames[it]]:
                if d['kind'] == 'weak entity':
                    itemcounts[filenames[it]]['weak_entity'] = 1
                else:
                    itemcounts[filenames[it]][d['kind']] = 1
            else:
                itemcounts[filenames[it]][d['kind']] += 1
        for d in data['relationships']:
            if d['kind'] not in itemcounts[filenames[it]]:
                if d['kind'] == 'identifying relationship':
                    itemcounts[filenames[it]]['identifying_relationship'] = 1
                else:
                    itemcounts[filenames[it]][d['kind']] = 1
            else:
                itemcounts[filenames[it]][d['kind']] += 1

    for item in itemcounts.keys():
        if 'entity' not in itemcounts[item].keys():
            itemcounts[item]['entity'] = 0
        if 'weak_entity' not in itemcounts[item].keys():
            itemcounts[item]['weak_entity'] = 0
        if 'relationship' not in itemcounts[item].keys():
            itemcounts[item]['relationship'] = 0
        if 'identifying_relationship' not in itemcounts[item].keys():
            itemcounts[item]['identifying_relationship'] = 0

    #returns the json dataset in python form, and a dictionary of dictionaries of counts per item type (format is dict[file][type]=counts)
    return dataset1, itemcounts

#word stemming. best to run on single word/phrase. returns the new version of that word
def stemming(data):
    ps = PorterStemmer()
    words = word_tokenize(data)
    news = []
    for w in words:
        news.append(ps.stem(w))
    word = ""
    for w in news:
        word += " "+w

    return word.strip()

#stop word removal. also based on single word/phrase. returns the new version of that word
def removestopwords(data):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(data)
    news = []
    for word in words:
        if word not in stop_words:
            news.append(word)
    word = ""
    for w in news:
        word += " "+w
    return word.strip()


#calculate rmse
#grades is of format [(name, grade), ...]
#results is of format [(name, grade), ...]
def rmse(grades, results):
    combos = []
    for r in results:
        for g in grades:
            if r[0] == g[0]:
                #combine both true score and prediction into one item
                combos.append((r[0], g[1], r[1]))
    # print(grades)
    # print(results)
    #calculate rmse
    se = 0
    for c in combos:
        se += (c[2] - c[1]) ** 2
    rmse = math.sqrt(se / len(combos))
    #calculate mae
    ae = 0
    for c in combos:
        ae += abs(c[2] - c[1])
    mae = ae / len(combos)
    return rmse, mae

#split dataset and perform testing of approaches
def splits(dataset, collection, grades, itemcounts, testset=None, testitems=None, testcollection=None, testing = True, seed=87, numt=70):
    trainingdata = []
    testingdata = []
    trainingdatanames = []
    trainingdataconts = []
    testingdatanames = []
    testingdataconts = []
    trainitemcount = {}
    testitemcount = {}

    nums = []
    if testing is True:
        random.seed(seed)
        nums = random.sample([d[0] for d in grades], numt)
        #print(nums)
        for i,doc in enumerate(dataset):
            parts = doc['erd_id'].split("_")

            if parts[0] in nums:

                trainingdata.append(doc)
                trainingdataconts.append(collection[i])
                trainingdatanames.append(parts[0])
                pts = doc['erd_id'].split(".")
                trainitemcount[parts[0]] = itemcounts[pts[0]]
        othernums = [d[0] for d in grades if d[0] not in nums]
        #print(othernums)
        for i,doc in enumerate(dataset):
            parts = doc['erd_id'].split("_")
            if parts[0] in othernums:
                testingdata.append(doc)
                testingdataconts.append(collection[i])

                testingdatanames.append(parts[0])

                pts = doc['erd_id'].split(".")
                testitemcount[parts[0]] = itemcounts[pts[0]]
    elif testing is False:
        for i,doc in enumerate(dataset):
            parts = doc['erd_id'].split("_")

            trainingdata.append(doc)
            trainingdataconts.append(collection[i])
            trainingdatanames.append(parts[0])
            pts = doc['erd_id'].split(".")
            trainitemcount[parts[0]] = itemcounts[pts[0]]


        #print(othernums)
        for i,doc in enumerate(testset):
            parts = doc['erd_id'].split("_")

            testingdata.append(doc)
            testingdataconts.append(testcollection[i])

            testingdatanames.append(parts[0])

            pts = doc['erd_id'].split(".")
            testitemcount[parts[0]] = testitems[pts[0]]



    """
    trainingdata is list of the json files in the training set
    testingddata is list of the json files in the testing set
    trainingdatanames is a list of the filenames of the training set
    testingdatanames is a list of the filenames of the testing set
    trainingdataconts is a list of the collection-style contents of each file in the training set
    testingdataconts is a list of the collection-style contents of each file in the testing set
    trainitemcount is a dictionary of counts per item type in the training set (formatted as the given itemcount)
    testitemcount is a dictionary of counts per item type in the testing set (formatted as the given itemcount)
    """
    return trainingdata, testingdata, trainingdatanames, testingdatanames, trainingdataconts, testingdataconts, trainitemcount, testitemcount



#sorts the grades correctly
def sortforcopying(results):
    re = sorted(results, key=lambda x: x[0], reverse=False)
    return re

#function to produce the results for an approach in a standard format suited for testing submission
def printresults(results1, results2, grades1, grades2, testnames, testing = True):
    print(results1)
    print(results2)
    if testing is True:
        rmse1, mae1 = rmse(grades1, results1)
        rmse2, mae2 = rmse(grades2, results2)
        print("RMSE for dataset 1: " + str(rmse1) + " MAE for dataset 1: " + str(mae1))
        print("RMSE for dataset 2: " + str(rmse2) + " MAE for dataset 1: " + str(mae2))
    one = sortforcopying(results1)
    two = sortforcopying(results2)

    y = 0
    z = 0
    testsetnames = testnames
    buildfile = []
    buildfile.append('ERD_No,dataset1_grade,dataset2_grade')
    for w in range(len(testsetnames)):
        if one[w + y][0] == two[w + z][0]:
            print(testsetnames[w] + "," + str(one[w + y][1]) + "," + str(two[w + z][1]))
            buildfile.append(testsetnames[w] + "," + str(one[w + y][1]) + "," + str(two[w + z][1]))
        elif one[w + y][0] != testsetnames[w]:
            print(testsetnames[w] + "," + "," + str(two[w + z][1]))
            buildfile.append(testsetnames[w] + "," + "," + str(two[w + z][1]))
            y -= 1
        elif two[w + z][0] != testsetnames[w]:
            print(testsetnames[w] + "," + str(one[w + y][1]) + ",")
            buildfile.append(testsetnames[w] + "," + str(one[w + y][1]) + ",")
            z -= 1

    return buildfile

#takes the results of each approach and creates the associated file filled with the predictions
def output_result_files(file1=None, file2=None, file3=range(1,5), file4=range(1,5), file5=range(1,5)):

    f = open('a1_tf.csv', "w")
    for line in file1:
        f.write(line +'\n')
    f.close()
    f = open('a2_emb.csv', "w")
    for line in file2:
        f.write(line +'\n')
    f.close()
    f = open('a3_graph2vec.csv', "w")
    for line in file3:
        f.write(str(line)+'\n')
    f.close()
    f = open('a4_custom_graph.csv', "w")
    for line in file4:
        f.write(line+'\n')
    f.close()
    f = open('ap5.csv', "w")
    for line in file5:
        f.write(str(line)+'\n')
    f.close()
    return


#research question 1 graph: arguments are the rmse of each approach
def rq1(ap1rmse, ap2remse, ap3rmse, ap4rmse, ap5rmse):
    categories = ['Approach 1', 'Approach 2', 'Approach 3', 'Approach 4', 'Approach 5']
    values = [ap1rmse[0], ap2remse[0], ap3rmse[0], ap4rmse[0], ap5rmse[0]]
    values2 = [ap1rmse[1], ap2remse[1], ap3rmse[1], ap4rmse[1], ap5rmse[1]]

    x = np.arange(len(categories))
    width = 0.35  # Width of the bars

    # Create the bar chart
    plt.bar(x - width/2, values, width, label='Dataset 1', color='skyblue')
    plt.bar(x + width/2, values2, width, label='Dataset 2', color='lightblue')
    # Add labels and title
    plt.xlabel('Approaches')
    plt.ylabel('RMSE')
    plt.title('Error Per Approach')
    #plt.set_xticklabels(categories)
    plt.xticks(x, categories)
    plt.legend()

    # Display the chart
    plt.tight_layout()
    plt.show()
    return


res11 = [('102', 77.44012454710985), ('103', 81.46771298711917), ('11', 79.09395816589502), ('12', 75.03070510950569), ('14', 88.73637419976757), ('18', 82.46112334434733), ('20', 76.27315691387973), ('23', 80.30244821276969), ('26', 79.10497965043923), ('27', 79.90520561582868), ('2', 75.5970988844787), ('30', 93.35561944850492), ('33', 75.13695922624008), ('41', 81.2161626451895), ('45', 88.08609683661707), ('48', 78.51431733372233), ('50', 77.73643076402136), ('53', 82.71931087497973), ('58', 74.78494325865896), ('60', 79.55065463670435), ('61', 86.06710056297159), ('69', 77.91108249461891), ('6', 78.22048447826863), ('70', 82.98481720125032), ('79', 77.41651844353213), ('84', 81.41746158624288), ('85', 83.66669093938138), ('86', 80.73586382606008), ('87', 77.44162591927652), ('94', 77.63446615569879), ('99', 76.9648845593841)]
res12 = [('102', 87.32905974414086), ('103', 96.71287849116378), ('11', 92.45609838695066), ('12', 84.59649655755456), ('14', 94.6314748752291), ('18', 96.80425830662975), ('20', 94.08849520646606), ('23', 95.37274126304692), ('26', 80.70863602061053), ('27', 86.01822348690715), ('2', 95.22724562995927), ('30', 94.71129150696937), ('33', 87.63953080079742), ('41', 92.08682490638293), ('45', 92.51874310822674), ('48', 87.90338628218737), ('50', 84.66990103883924), ('53', 88.68278278775139), ('58', 90.24912609809822), ('60', 95.47910727911594), ('61', 85.25084883598569), ('69', 91.10005589517604), ('6', 92.5223803665187), ('70', 83.2256243897828), ('79', 85.10423574550622), ('84', 91.81318543237518), ('85', 88.73713712638465), ('86', 92.89080389931465), ('87', 91.78735479206227), ('94', 94.57528532037871), ('99', 92.5433197201446)]
res21 = [('102', 82.93541626831616), ('103', 82.66996914374553), ('11', 76.56134161568085), ('12', 75.19688944033159), ('14', 81.59515316569767), ('18', 88.00012791530253), ('20', 78.93450244707327), ('23', 73.33145320144348), ('26', 78.93707388270589), ('27', 83.99608697108862), ('2', 80.53689367789583), ('30', 75.72710712774082), ('33', 78.9362104038384), ('41', 80.80125198727403), ('45', 75.7277681196747), ('48', 74.12579771718308), ('50', 78.39793715979772), ('53', 78.66713738329251), ('58', 78.9381286911049), ('60', 79.73706506371747), ('61', 78.12763481411773), ('69', 80.8024108543079), ('6', 78.93623552850444), ('70', 81.86659467797118), ('79', 77.59881707590236), ('84', 84.80051763218688), ('85', 82.93705247949997), ('86', 81.07056612811107), ('87', 74.11763874158079), ('94', 80.53653065597254), ('99', 81.07036629590633)]
res22 = [('102', 73.93882342555402), ('103', 79.20183727178868), ('11', 80.26611553185681), ('12', 75.46737164574289), ('14', 75.73213063491426), ('18', 78.66557722335763), ('20', 77.86565788413223), ('23', 76.5344310688097), ('26', 78.66605027103755), ('27', 79.73178697043825), ('2', 77.5998671653208), ('30', 83.70216565040552), ('33', 77.86685029856717), ('41', 79.19697578680689), ('45', 79.73524623200824), ('48', 78.13469820951268), ('50', 77.33353198111097), ('53', 76.2668551211752), ('58', 72.80000097415278), ('60', 76.26797764964967), ('61', 79.73278801366618), ('69', 77.33373541689458), ('6', 84.23916555354567), ('70', 75.20315131495101), ('79', 73.86844655062075), ('84', 78.13346485166885), ('85', 75.7336280518003), ('86', 75.46871980061759), ('87', 76.53554426950704), ('94', 78.13478123279775), ('99', 77.59754666979391)]
res31 = [('102', 79.0296575339218), ('103', 79.77330351230917), ('11', 78.7871492471576), ('12', 77.89856502475547), ('14', 82.39031079273397), ('18', 80.13364314678034), ('20', 78.75444033359025), ('23', 79.01339050639703), ('26', 79.22330217799124), ('27', 78.7484386568915), ('2', 77.72678120515873), ('30', 85.9939812997968), ('33', 79.45522581176363), ('41', 78.36610336366368), ('45', 79.71086543063538), ('48', 78.24836701300785), ('50', 79.21148205298555), ('53', 79.35743635603166), ('58', 77.94657589869455), ('60', 76.49184212773403), ('61', 81.85517762917821), ('69', 78.65689526341438), ('6', 78.96407787778703), ('70', 79.84022025274399), ('79', 78.7832913152749), ('84', 77.9473766900648), ('85', 79.15528640965636), ('86', 81.50425939095412), ('87', 77.56402351867737), ('94', 79.4283244259729), ('99', 78.01270832553787)]
res32 = [('102', 86.97303116206267), ('103', 92.05878027976837), ('11', 89.39279685774703), ('12', 86.626907239179), ('14', 92.65314273838757), ('18', 86.69993362984094), ('20', 90.38934238035418), ('23', 93.7437409952677), ('26', 85.11916949709588), ('27', 84.67496346151852), ('2', 88.28697130293376), ('30', 88.73354264011462), ('33', 91.03404133652317), ('41', 88.76171763066144), ('45', 88.1194073436437), ('48', 87.9388006606684), ('50', 86.94612595033091), ('53', 89.73598485043205), ('58', 89.97662262510948), ('60', 90.10724291305897), ('61', 86.78440006718684), ('69', 89.35870686486453), ('6', 89.6266702138301), ('70', 84.94815089803564), ('79', 81.89193572368455), ('84', 91.12064738986122), ('85', 85.42317120567387), ('86', 90.36610064353046), ('87', 87.06393334928671), ('94', 91.23859555765638), ('99', 92.6781635362697)]
res41 = [('102', 81.33115375424822), ('103', 76.66667451243816), ('11', 82.16118239698801), ('12', 74.83225515884367), ('14', 80.8875880641865), ('18', 82.12546739003342), ('20', 78.66533277071612), ('23', 83.92395956232654), ('26', 79.96057339347398), ('27', 75.85555705249064), ('2', 80.07827354255538), ('30', 81.74471050021717), ('33', 78.16253827488893), ('41', 80.50693402749967), ('45', 82.25666846012584), ('48', 80.80809466636208), ('50', 81.07103489910409), ('53', 77.35901323727386), ('58', 76.17189729569327), ('60', 77.80440068946814), ('61', 81.12441178106955), ('69', 81.37067959373587), ('6', 83.32747461729316), ('70', 80.99373989523646), ('79', 74.15801039393415), ('84', 79.49358936894849), ('85', 83.0073867844772), ('86', 80.32655498497591), ('87', 78.39178194434754), ('94', 77.27152922508246), ('99', 77.15722905559795)]
res42 = [('102', 92.99481505309448), ('103', 88.38543535195767), ('11', 95.02557779399089), ('12', 87.41113504107726), ('14', 96.00077491913099), ('18', 87.49769714721285), ('20', 91.44438486337829), ('23', 96.00041103055362), ('26', 90.49170543735634), ('27', 97.0004772718566), ('2', 81.07219501045049), ('30', 90.6106500021798), ('33', 80.99164132553938), ('41', 96.47300599900052), ('45', 96.00041103055362), ('48', 96.01465297779377), ('50', 87.48413063002813), ('53', 93.98804928379853), ('58', 84.47353130788562), ('60', 87.34434639069677), ('61', 93.50120791654943), ('69', 89.05305567256742), ('6', 95.50309745354909), ('70', 97.49998038943791), ('79', 88.95133714539227), ('84', 93.98756429410706), ('85', 91.0145795111743), ('86', 92.99483588772515), ('87', 92.47324021275013), ('94', 88.9969158849075), ('99', 95.00206850635732)]
res51 = [('102', 79.74159369632156), ('103', 78.8867318152662), ('11', 80.68417095064436), ('12', 74.93109914173758), ('14', 84.21810711361617), ('18', 82.45479548010691), ('20', 77.66655944532329), ('23', 82.05011504688386), ('26', 79.56623048257815), ('27', 77.8280799499826), ('2', 78.20299705137256), ('30', 86.44520632371976), ('33', 76.91466261132379), ('41', 80.81278084331737), ('45', 84.49779419037601), ('48', 79.6227324201885), ('50', 79.57786036005294), ('53', 79.66137830611169), ('58', 75.6792642903516), ('60', 78.60101769234679), ('61', 83.1100818163157), ('69', 79.89203984930104), ('6', 81.02932544688218), ('70', 81.85934467232437), ('79', 75.64710086743938), ('84', 80.47399780826498), ('85', 83.27847348527331), ('86', 80.52436941492213), ('87', 77.8522682200454), ('94', 77.53253059700751), ('99', 77.20386357281936)]
res52 = [('102', 89.98467687729428), ('103', 91.5997836075496), ('11', 93.459133033536), ('12', 85.83804458264886), ('14', 94.7640606480584), ('18', 91.13079751169755), ('20', 92.11729079682371), ('23', 95.10328671140233), ('26', 85.98216061580628), ('27', 91.8017900546316), ('2', 86.9300281137348), ('30', 92.11207190822182), ('33', 83.70188156218444), ('41', 94.05755059440565), ('45', 93.99847349649937), ('48', 92.01022372494401), ('50', 85.96954304386327), ('53', 91.17495550484077), ('58', 86.53448571518794), ('60', 90.41523354199029), ('61', 89.5728329833493), ('69', 89.53618767448006), ('6', 93.87633188284528), ('70', 90.75161479289497), ('79', 86.83467931198683), ('84', 92.55521110585762), ('85', 89.55430459399086), ('86', 92.3806526032102), ('87', 91.66227437384455), ('94', 90.99739631891211), ('99', 93.40035479850114)]




#research question 2: arguments are everything needed to perform splits()
def rq2(dataset1, collectionlist1, ds1grades, itemcounts1, dataset1t, itemcounts1t, collectionlist1t, dataset2, collectionlist2, ds2grades, itemcounts2, dataset2t, itemcounts2t, collectionlist2t, gradespath):
    testing = True
    numt = [10,20,30,40,50,60,70,80,90]
    ks = [1,2,3,4,5,10,20,50]
    pairrmse = {}
    for num in numt:
        for k in ks:
            print(num, k)
            trainingdata1, testingdata1, trainingdatanames1, testingdatanames1, trainingdataconts1, testingdataconts1, trainitemcount1, testitemcount1 = splits(
                dataset1, collectionlist1, ds1grades, itemcounts1, dataset1t, itemcounts1t, collectionlist1t, testing=testing, numt=num)
            trainingdata2, testingdata2, trainingdatanames2, testingdatanames2, trainingdataconts2, testingdataconts2, trainitemcount2, testitemcount2 = splits(
                dataset2, collectionlist2, ds2grades, itemcounts2, dataset2t, itemcounts2t, collectionlist2t, testing=testing, numt=num)

            #PUT CHOSEN APPROACH HERE


            results31 = approach3(trainingdata1, testingdata1, trainingdataconts1, testingdataconts1, testingdatanames1,
                                  gradespath)
            results32 = approach3(trainingdata2, testingdata2, trainingdataconts2, testingdataconts2, testingdatanames2,
                                  gradespath)

            rmse1, mae1 = rmse(ds1grades, results31)
            rmse2, mae2 = rmse(ds2grades, results32)
            rmseavg = (rmse1+rmse2)/2
            pairrmse[(num, k)] = rmseavg

    # Suppose pairrmse[(x,y)] = value
    xs = sorted({k[0] for k in pairrmse})
    ys = sorted({k[1] for k in pairrmse})

    # Map original coordinates → compact grid indices
    x_to_idx = {x: i for i, x in enumerate(xs)}
    y_to_idx = {y: i for i, y in enumerate(ys)}

    # Create tightly sized array
    heatmap_data = np.full((len(ys), len(xs)), np.nan)

    # Fill it
    for (x, y), value in pairrmse.items():
        heatmap_data[y_to_idx[y], x_to_idx[x]] = value

    # 3. Create the heatmap using imshow
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, cmap='viridis', origin='lower')
    plt.colorbar(label='RMSE')

    plt.xticks(ticks=range(len(xs)), labels=xs)
    plt.yticks(ticks=range(len(ys)), labels=ys)

    plt.title('Heatmap from (T,K) pairs')
    plt.xlabel('# of grades (T)')
    plt.ylabel('# of neighbors for knn (K)')

    # # Optional: Add annotations for values in each cell
    # for (x, y), value in data_dict.items():
    #     plt.text(y, x, f'{value}', ha='center', va='center', color='white')

    plt.grid(False)  # Remove grid lines for a cleaner heatmap
    plt.show()
    return


#research question 3: arguments are everything to perform splits()
def rq3(dataset1, collectionlist1, ds1grades, itemcounts1, dataset1t, itemcounts1t, collectionlist1t, dataset2, collectionlist2, ds2grades, itemcounts2, dataset2t, itemcounts2t, collectionlist2t):
    testing = True
    trainingdata1, testingdata1, trainingdatanames1, testingdatanames1, trainingdataconts1, testingdataconts1, trainitemcount1, testitemcount1 = splits(
        dataset1, collectionlist1, ds1grades, itemcounts1, dataset1t, itemcounts1t, collectionlist1t, testing=testing)
    trainingdata2, testingdata2, trainingdatanames2, testingdatanames2, trainingdataconts2, testingdataconts2, trainitemcount2, testitemcount2 = splits(
        dataset2, collectionlist2, ds2grades, itemcounts2, dataset2t, itemcounts2t, collectionlist2t, testing=testing)

    # PUT CHOSEN APPROACH HERE
    params = {'A': 0.008097658971528523, 'B': 0.013945646059937452, 'C': 0.21883992806812,
              'D': 0.7591167669004141, 'T': 0.06597323666238664, 'U': 0.00762455521148119,
              'V': 0.0048008844641487525, 'X': 0.012301884649412755, 'Y': 0.024146228140775598,
              'Z': 0.8851532108717951, 'alpha': 0.006497542159462399, 'beta': 0.10882968354314841,
              'lambda': 0.8846727742973891, 'entity_type_c': 0.5743956407590298,
              'rel_type_c': 0.40550007128340027, 'K': 8}
    k = params['K']
    results1 = approach4_1(dataset1, collectionlist1, testingdata1, trainingdata1, testingdataconts1,
                         trainingdataconts1, trainingdatanames1, testingdatanames1, trainitemcount1,
                         testitemcount1, ds1grades, params, k=k)
    results2 = approach4_1(dataset2, collectionlist2, testingdata2, trainingdata2, testingdataconts2,
                         trainingdataconts2, trainingdatanames2, testingdatanames2, trainitemcount2,
                         testitemcount2, ds2grades, params, k=k)

    ds1close = []
    for r in results1:
        grade1 = 0
        grade2 = 0
        for g in ds1grades:
            if r[0]==g[0]:
                grade1 = g[1]
            if r[1]==g[0]:
                grade2 = g[1]
        if grade1 != 0 or grade2 != 0:
            sim = 1- (abs(grade1-grade2)/max(abs(grade1), abs(grade2)))
        else:
            sim = 0
        ds1close.append((r[0],r[1],sim))

    ds2close = []
    for r in results2:
        grade1 = 0
        grade2 = 0
        for g in ds2grades:
            if r[0] == g[0]:
                grade1 = g[1]
            if r[1] == g[0]:
                grade2 = g[1]
        if grade1 != 0 or grade2 != 0:
            sim = 1 - (abs(grade1 - grade2) / max(abs(grade1), abs(grade2)))
        else:
            sim = 0
        ds2close.append((r[0], r[1], sim))
    ds1close = sorted(ds1close, key= lambda x: x[2], reverse=True)
    ds2close = sorted(ds2close, key= lambda x: x[2], reverse=True)

    print(ds1close)
    print(ds2close)
    return




