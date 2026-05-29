import os
import json
import utilities as u
import numpy as np


def cos_sim(erd1, erd2):
    dot_prod = 0
    for key in erd1.keys():
        dot_prod += erd1[key] * erd2[key]
    mag1 = np.linalg.norm(erd1)
    mag2 = np.linalg.norm(erd2)

    return dot_prod/ (mag1 * mag2)


path = "C:/Users/learj/PycharmProjects/pythonProject/cs473/project/for_students/for_students/Dataset1"
ds1 = os.listdir(path)
# print(ds1)
ds1files = []
filenames = []
for i in ds1:
    fi, ex = os.path.splitext(i)
    if ex.lower() == '.json':
        ds1files.append(i)
        filenames.append(fi)
dataset1 = []
itemcounts = {}
# print(ds1files)
for it, i in enumerate(ds1files):
    f = open(path + "/" + i, "r")
    data = json.load(f)
    dataset1.append(data)

    itemcounts[filenames[it]] = {}
    for d in data['entities']:
        kind = u.replaceword(d["kind"])
        if kind not in itemcounts[filenames[it]]:
            if kind == 'weak_entity':
                itemcounts[filenames[it]]['weak_entity'] = 1
            else:
                itemcounts[filenames[it]][kind] = 1
        else:
            itemcounts[filenames[it]][kind] += 1

    itemcounts[filenames[it]]["relationship_attributes"] = 0
    itemcounts[filenames[it]]["binary_relationship"] = 0
    itemcounts[filenames[it]]["n_way_relationship"] = 0
    itemcounts[filenames[it]]["binary_identifying_relationship"] = 0
    itemcounts[filenames[it]]["n_way_identifying_relationship"] = 0
    for d in data['relationships']:
        kind = u.replaceword(d['kind'])
        n = len(d['involved_entities'])
        itemcounts[filenames[it]]["relationship_attributes"] += len(d['attributes'])

        if kind not in itemcounts[filenames[it]]:
            if kind == 'identifying_relationship':
                itemcounts[filenames[it]][kind] = 1

            else:
                itemcounts[filenames[it]][kind] = 1

        else:
            itemcounts[filenames[it]][kind] += 1

        if n == 2 and n is not None:
            # print(kind)
            print(f"binary_{kind}")
            itemcounts[filenames[it]][f"binary_{kind}"] += 1
        elif n > 2 and n is not None:
            # print(kind)
            print(f"n_way_{kind}")
            itemcounts[filenames[it]][f"n_way_{kind}"] += 1

for item in itemcounts.keys():
    if 'entity' not in itemcounts[item].keys():
        itemcounts[item]['entity'] = 0
    if 'weak_entity' not in itemcounts[item].keys():
        itemcounts[item]['weak_entity'] = 0
    if 'relationship' not in itemcounts[item].keys():
        itemcounts[item]['relationship'] = 0
    if 'identifying_relationship' not in itemcounts[item].keys():
        itemcounts[item]['identifying_relationship'] = 0

print(itemcounts)