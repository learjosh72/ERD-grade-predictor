import random

import optuna
import numpy as np
from utilities import clean_dataset, normalize_data, splits, rmse
from approach1 import approach1
from approach4 import approach4
def prepare_data():
    ds1path = "C:/Users/learj/PycharmProjects/pythonProject/cs473/project/for_students/for_students/Dataset1"
    ds2path = "C:/Users/learj/PycharmProjects/pythonProject/cs473/project/for_students/for_students/Dataset2"
    ds1tpath = "C:/Users/learj/PycharmProjects/pythonProject/cs473/project/for_testing/for_testing/Dataset1"
    ds2tpath = "C:/Users/learj/PycharmProjects/pythonProject/cs473/project/for_testing/for_testing/Dataset2"

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

    dataset1, itemcounts1 = clean_dataset(
        ds1path)
    dataset2, itemcounts2 = clean_dataset(
        ds2path)
    # dataset1t, itemcounts1t = clean_dataset(
    #    "C:/Users/learj/PycharmProjects/pythonProject/cs473/project/for_testing/for_testing/Dataset1")
    # dataset2t, itemcounts2t = clean_dataset(
    #    "C:/Users/learj/PycharmProjects/pythonProject/cs473/project/for_testing/for_testing/Dataset2")

    dataset1, collectionlist1 = normalize_data(dataset1)
    dataset2, collectionlist2 = normalize_data(dataset2)
    # dataset1t, collectionlist1t = normalize_data(dataset1t)
    # dataset2t, collectionlist2t = normalize_data(dataset2t)
    return dataset1, dataset2, itemcounts1, itemcounts2, collectionlist1, collectionlist2, ds1grades, ds2grades

dataset1, dataset2, itemcounts1, itemcounts2, collectionlist1, collectionlist2, ds1grades, ds2grades = prepare_data()

def evaluate_cv(params):

    li = random.sample(range(1,101), 5)
    rmsesum = 0
    for i in range(len(li)):
        testing = True
        trainingdata1, testingdata1, trainingdatanames1, testingdatanames1, trainingdataconts1, testingdataconts1, trainitemcount1, testitemcount1 = splits(
            dataset1, collectionlist1, ds1grades, itemcounts1, testing=testing, seed=li[i], numt=80)
        trainingdata2, testingdata2, trainingdatanames2, testingdatanames2, trainingdataconts2, testingdataconts2, trainitemcount2, testitemcount2 = splits(
            dataset2, collectionlist2, ds2grades, itemcounts2, testing=testing, seed=li[i], numt=80)

        k = params['K']

        results1 = approach1(trainingdata1, testingdata1, trainingdataconts1, testingdataconts1, trainingdatanames1,
                              testingdatanames1, trainitemcount1, testitemcount1, ds1grades, params, k=k)
        results2 = approach1(trainingdata2, testingdata2, trainingdataconts2, testingdataconts2, trainingdatanames2,
                              testingdatanames2, trainitemcount2, testitemcount2, ds2grades, params, k=k)

        rmse1, mae1 = rmse(ds1grades,results1)
        rmse2, mae2 = rmse(ds2grades,results2)

        rmsesum+=((rmse1+rmse2)/2)
    return rmsesum/len(li)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def sample_simplex(trial, n, name):
    raw = np.array([trial.suggest_float(f"{name}_{i}", 0, 4) for i in range(n)])
    return softmax(raw)

def objective(trial):

    # ----- Entity weights -----
    #ent = sample_simplex(trial, 4, "ent")
    raw = np.array([trial.suggest_float(f"ent_{i}", 0, 4) for i in range(4)])
    A, B, C, D = raw



    # ----- Other scalars -----
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    beta = trial.suggest_float("beta", 0.0, 100.0)

    K = trial.suggest_int("K", 3, 6)

    # ----- Pack parameters -----
    params = {
        "A": A, "B": B, "C": C, "D": D,
        "alpha": alpha, "beta": beta,
        "K": K
    }

    # ----- Your cross-validation scoring -----
    rmse = evaluate_cv(params)   # <-- you implement this

    return rmse

#study = optuna.create_study(direction="minimize")
#study.optimize(objective, n_trials=1000)  # 300–600 recommended

#print("Best params:", study.best_params)
#print("Best RMSE:", study.best_value)


# ps = {'ent_0': -3.270390450025565, 'ent_1': -2.7267981049181116, 'ent_2': 0.02637508642920494, 'ent_3': 1.270190154576396, 'rel_0': -1.0878543240072063, 'rel_1': -3.245729490807301, 'rel_2': -3.708303314375085, 'rel_3': -2.767351004508507, 'rel_4': -2.0929752954253287, 'rel_5': 1.5086572709747574, 'erd_0': -2.7286066499010895, 'erd_1': 0.08975349791447229, 'erd_2': 2.185187204097907, 'entity_type_c': 0.5743956407590298, 'rel_type_c': 0.40550007128340027, 'K': 8}
ps = {'ent_0': 2.8573974765927592, 'ent_1': 0.6712578444597177, 'ent_2': 3.601158590055459, 'ent_3': 3.000557922549613, 'alpha': 0.9991572245411128, 'beta': 0.035051644786343844, 'K': 5}

#
ent_raw = np.array([ps[f'ent_{i}'] for i in range(4)])
A,B,C,D = ent_raw
alpha = ps['alpha']
beta = ps['beta']
K = ps['K']
# rel_raw = np.array([ps[f'rel_{i}'] for i in range(6)])
# T, U, V, X, Y, Z = softmax(rel_raw)
# erd_raw = np.array([ps[f'erd_{i}'] for i in range(3)])
# alpha, beta, lam = softmax(erd_raw)
# entity_type_c = ps['entity_type_c']
# rel_type_c = ps['rel_type_c']
# K = ps['K']
params = {
        "A": A, "B": B, "C": C, "D": D,
        "alpha": alpha, "beta": beta,
        "K": K
    }
print(params)
