import random

import optuna
import numpy as np
from utilities import clean_dataset, normalize_data, splits, rmse
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

        results1 = approach4(dataset1, collectionlist1, testingdata1, trainingdata1, testingdataconts1,trainingdataconts1, trainingdatanames1, testingdatanames1, trainitemcount1, testitemcount1, ds1grades, params=params, k=k)
        results2 = approach4(dataset2, collectionlist2, testingdata2, trainingdata2, testingdataconts2,trainingdataconts2, trainingdatanames2, testingdatanames2, trainitemcount2, testitemcount2, ds2grades, params=params, k=k)

        rmse1, mae1 = rmse(ds1grades,results1)
        rmse2, mae2 = rmse(ds2grades,results2)

        rmsesum+=((rmse1+rmse2)/2)
    return rmsesum/len(li)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def sample_simplex(trial, n, name):
    raw = np.array([trial.suggest_float(f"{name}_{i}", -4, 4) for i in range(n)])
    return softmax(raw)

def objective(trial):

    # ----- Entity weights -----
    ent = sample_simplex(trial, 4, "ent")
    A, B, C, D = ent

    # ----- Relationship weights -----
    # rel = sample_simplex(trial, 6, "rel")
    # T, U, V, X, Y, Z = rel
    base = trial.suggest_float("base", -3, 3)
    d1 = trial.suggest_float("d1", 0.01, 5.0)
    d2 = trial.suggest_float("d2", 0.01, 5.0)
    d3 = trial.suggest_float("d3", 0.01, 5.0)
    d4 = trial.suggest_float("d4", 0.01, 5.0)

    Z = base
    Y = Z +d1
    X = Y + d2
    T = X + d3
    U = T + d4
    V = U

    Z,Y,X,T,U,V = softmax(np.array([Z,Y,X,T,U,V]))

    # ----- ERD-level weights -----
    erd_w = sample_simplex(trial, 3, "erd")
    alpha, beta, lam = erd_w

    # ----- Other scalars -----
    entity_type_c = trial.suggest_float("entity_type_c", 0.0, 1.0)
    rel_type_c = trial.suggest_float("rel_type_c", 0.0, 1.0)

    K = trial.suggest_int("K", 1, 10)

    # ----- Pack parameters -----
    params = {
        "A": A, "B": B, "C": C, "D": D,
        "T": T, "U": U, "V": V, "X": X, "Y": Y, "Z": Z,
        "alpha": alpha, "beta": beta, "lambda": lam,
        "entity_type_c": entity_type_c,
        "rel_type_c": rel_type_c,
        "K": K
    }

    # ----- Your cross-validation scoring -----
    rmse = evaluate_cv(params)   # <-- you implement this

    return rmse

#study = optuna.create_study(direction="minimize")
#study.optimize(objective, n_trials=300)  # 300–600 recommended

#print("Best params:", study.best_params)
#print("Best RMSE:", study.best_value)


ps = {'ent_0': 1.5239724018559564, 'ent_1': 3.4634348679195273, 'ent_2': -0.11395392406380006, 'ent_3': 1.5126655412795114, 'base': 2.456862326443152, 'd1': 1.933153449782964, 'd2': 1.8999505330681312, 'd3': 1.8057267134134367, 'd4': 3.155284011293461, 'erd_0': -3.5323538641233956, 'erd_1': -3.374720035904209, 'erd_2': 1.3785470961524091, 'entity_type_c': 0.39251387917953096, 'rel_type_c': 0.11398878584290015, 'K': 7}
# Best RMSE: 6.9280482452765835


#ps = {'ent_0': -3.270390450025565, 'ent_1': -2.7267981049181116, 'ent_2': 0.02637508642920494, 'ent_3': 1.270190154576396, 'rel_0': -1.0878543240072063, 'rel_1': -3.245729490807301, 'rel_2': -3.708303314375085, 'rel_3': -2.767351004508507, 'rel_4': -2.0929752954253287, 'rel_5': 1.5086572709747574, 'erd_0': -2.7286066499010895, 'erd_1': 0.08975349791447229, 'erd_2': 2.185187204097907, 'entity_type_c': 0.5743956407590298, 'rel_type_c': 0.40550007128340027, 'K': 8}
# ps = {'ent_0': 0.7728499978905088, 'ent_1': 2.1758478658611247, 'ent_2': -3.1190816547984594, 'ent_3': 0.9387052423191757, 'base': -0.7321139886084365, 'd1': 3.4634388341320057, 'd2': 3.052011057234786, 'd3': 0.3190887636367772, 'd4': 0.5771403179422677, 'd5': 2.866883598583566, 'erd_0': -0.5065596975882236, 'erd_1': -2.0350887017832897, 'erd_2': 2.342072992907162, 'entity_type_c': 0.06301128839652652, 'rel_type_c': 0.24769027081285427, 'K': 8}
ent_raw = np.array([ps[f'ent_{i}'] for i in range(4)])
A,B,C,D = softmax(ent_raw)
#
z = ps['base']
y = z + ps['d1']
x = y + ps['d2']
t = x + ps['d3']
u = t + ps['d4']
v = u
rels = np.array([t,u,v,x,y,z])
#rel_raw = np.array([ps[f'rel_{i}'] for i in range(6)])
T, U, V, X, Y, Z = softmax(rels)
erd_raw = np.array([ps[f'erd_{i}'] for i in range(3)])
alpha, beta, lam = softmax(erd_raw)
entity_type_c = ps['entity_type_c']
rel_type_c = ps['rel_type_c']
K = ps['K']
params = {
        "A": A, "B": B, "C": C, "D": D,
        "T": T, "U": U, "V": V, "X": X, "Y": Y, "Z": Z,
        "alpha": alpha, "beta": beta, "lambda": lam,
        "entity_type_c": entity_type_c,
        "rel_type_c": rel_type_c,
        "K": K
    }
print(params)
