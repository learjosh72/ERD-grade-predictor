"""approach3.py

Performs KNN regression of ERDs using a combination of Graph2Vec and TF.IDF
similarity measures.

Usage: approach3.py <dataset1_path> <dataset2_path> <grades_path>
"""

import csv
from erdvec import ERDvec
import gensim
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import pprint
import random
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import sys
import time
import utilities as util

_NUM_ARGS = 4
_NUM_PARTITIONS = 5
_MIN_ALPHA = 0.0
_MAX_ALPHA = 1.0
_NUM_ALPHA_STEPS = 101
_ALPHA_STEP = (_MAX_ALPHA - _MIN_ALPHA) / (_NUM_ALPHA_STEPS - 1)
_MIN_K = 1
_NUM_K_STEPS = 100


def _validate_args() -> None:
    """Checks if the number of command line args is correct. If it is, then
    it returns. Otherwise, it displays an error message to stderr and exits with
    status 1."""
    if len(sys.argv) == _NUM_ARGS:
        return
    sys.stderr.write('Usage: %s <dataset1_path> <dataset2_path> <grades_path>\n' % \
                     sys.argv[0])
    exit(1)


def _create_partitions(dataset: list) -> list:
    """Creates the list of ranges for the partitons.

    Each range is for one testing round. The first element of the range is
    inclusive, the last is exclusive."""
    parition_size = int(len(dataset) / _NUM_PARTITIONS)
    partitions = []
    start = 0
    for i in range(_NUM_PARTITIONS - 1):
        end = start + parition_size
        partitions.append((start, end))
        start = end
    partitions.append((start, len(dataset)))
    return partitions


def _split_dataset(valid_range: tuple, dataset: list) -> tuple:
    """Splits the dataset into training and validation datasets.

    The validation dataset is specified by the given validation range. The
    training dataset is all other data elements.
    """
    valid_start = valid_range[0]
    valid_end = valid_range[1]
    train_set = dataset[:valid_start] + dataset[valid_end:]
    valid_set = dataset[valid_start: valid_end]
    return train_set, valid_set


def _create_erd_map(dataset: list) -> dict:
    """Creates a dict mapping indicies to the erd numbers.
    """
    erd_map = {}
    for i, erd in enumerate(dataset):
        fname = erd['erd_id']
        end = fname.find('_')
        erd_no = int(fname[:end])
        erd_map[i] = erd_no
    return erd_map


def _parse_dataset(dataset: list) -> tuple:
    """Takes the dataset, normalizes it, and generates a BoW for every ERD.

    The dataset is a list of JSON data in the form of dicts.
    The normalized dataset, the dictionary, and the BoWs are returned."""
    dataset, docs = util.normalize_data(dataset)
    dictionary = Dictionary(docs)
    doc_bows = [dictionary.doc2bow(doc) for doc in docs]
    return dataset, dictionary, doc_bows


def _get_tfidf_vector(tfidf_model: TfidfModel, dictionary: Dictionary, \
                      doc_bow: list) -> np.ndarray:
    """Converts a doc BoW into a vector.

    The vector is represented using a list of the TF.IDF values.
    """
    doc_tfidf = tfidf_model[doc_bow]
    vector = np.zeros(len(dictionary))
    for component in doc_tfidf:
        i = component[0]
        tfidf = component[1]
        vector[i] = tfidf
    return vector


def _generate_tfidf_matrix(tfidf_model: TfidfModel, \
                           dictionary: Dictionary, doc_bows: list) -> np.ndarray:
    """Creates and returns a matrix of TF.IDF vectors."""
    tfidf_matrix = np.zeros((len(doc_bows), len(dictionary)))
    for i, doc_bow in enumerate(doc_bows):
        vector = _get_tfidf_vector(tfidf_model, dictionary, doc_bow)
        tfidf_matrix[i] = vector
    return tfidf_matrix


def _calc_similarities(g2v_similarities: np.ndarray, tfidf_similarities: np.ndarray, \
                       alpha: float) -> np.ndarray:
    """Calculates the combined Graph2Vec and TF.IDF similarities.

    The combined similarity uses the following formula:
        similarity = alpha * graph2vec_sim + (1.0 - alpha) * tfidf_sim
    where alpha is a hyperparameter in the range [0.0, 1.0].

    The results are returned in a NxN ndarray, where N is the number of
    documents.
    """
    return alpha * g2v_similarities + (1.0 - alpha) * tfidf_similarities


def _calc_nearest_neighbor_mat(similarity_matrix: np.ndarray) -> np.ndarray:
    """Returns the nearest neighbor matrix.

    Each row represents the document, each column represents the ranking of
    similarity, and each element represents the index of the document at that
    ranking for that row's document."""
    return np.argsort(similarity_matrix, axis=1)[::-1]


def _find_k_nearest_neighbors(k: int, nn_matrix: np.ndarray) \
        -> np.ndarray:
    """Calculates the k nearest neighbors for each document."""
    knn_matrix = np.zeros((nn_matrix.shape[0], k), dtype=int)
    for i in range(nn_matrix.shape[0]):
        j = 0
        for rank in range(nn_matrix.shape[1]):
            docnum = nn_matrix[i][rank]
            if docnum == i:
                continue
            knn_matrix[i][j] = docnum
            j += 1
            if j >= k:
                break
    return knn_matrix


def _read_grades(grades_path: str) -> dict:
    """Reads the grades data from the file at the given grades path, and returns
    them in a dict.

    The dict has a key indicating the ERD number, another key indicating the
    dataset number, and the value indicating the grade."""
    grades_dict = {}
    with open(grades_path, mode='r') as grades_file:
        csv_reader = csv.reader(grades_file)
        is_first_line_read = False
        for line in csv_reader:
            if not is_first_line_read:
                is_first_line_read = True
                continue
            erd_id = int(line[0])
            grade1 = float(line[1])
            grade2 = float(line[2])
            grades_dict[erd_id] = {1: grade1, 2: grade2}
    return grades_dict


def _predict_grades(knn_matrix: np.ndarray, similarity_matrix: np.ndarray, \
                    valid_erd_map: dict, train_erd_map: dict, dataset_no: int, \
                    grades_dict: dict) -> np.ndarray:
    """Based on the k nearest neighbors, predict the grades of the validation
    set."""
    predicted_grades = np.zeros(knn_matrix.shape[0])
    for i in range(knn_matrix.shape[0]):
        weighted_sum = 0.0
        total_similarity = 0.0
        for j in range(knn_matrix.shape[1]):
            docnum = knn_matrix[i][j]
            similarity = similarity_matrix[i][docnum]
            erd_no = train_erd_map[docnum]
            grade = grades_dict[erd_no][dataset_no]
            weighted_sum += similarity * grade
            total_similarity += similarity
        predicted_grades[i] = weighted_sum / total_similarity
    return predicted_grades






def _root_mean_square_error(predicted_grades: np.ndarray, valid_erd_map: dict, \
                            dataset_no: int, grades_dict: dict) -> float:
    """Calculates the root mean square error between the predicted and actual
    grades."""
    square_sums = 0.0
    for i, prediction in enumerate(predicted_grades):
        erd_no = valid_erd_map[i]
        grade = grades_dict[erd_no][dataset_no]
        square_sums += (prediction - grade) ** 2.0
    return math.sqrt(square_sums / len(predicted_grades))


def _evaluate(dataset: list, dataset_no: int, valid_range: tuple, \
              alpha: float, k: int, grades_dict: dict) -> float:
    """Performs one round of validation, given the dataset, the dataset number,
    the validation data range, the hyperparameters alpha and k, and the grades
    dictionary.

    Returns the root mean square error of the model's predictions.
    """
    # Splitting data
    train_set, valid_set = _split_dataset(valid_range, dataset)
    # Mapping document numbers to ERD numbers
    train_erd_map = _create_erd_map(train_set)
    valid_erd_map = _create_erd_map(valid_set)
    # Train Graph2Vector
    train_g2v_matrix = ERDvec.make_graph2vec_matrix(train_set)
    valid_g2v_matrix = ERDvec.make_graph2vec_matrix(valid_set)
    g2v_similarities = cosine_similarity(valid_g2v_matrix, train_g2v_matrix)
    # Create dictionary and BoWs
    dataset, dictionary, doc_bows = _parse_dataset(dataset)
    train_bows, valid_bows = _split_dataset(valid_range, doc_bows)
    # Train TF.IDF
    tfidf_model = TfidfModel(train_bows)
    train_tfidf_mat = _generate_tfidf_matrix(tfidf_model, dictionary, \
                                             train_bows)
    valid_tfidf_mat = _generate_tfidf_matrix(tfidf_model, dictionary, \
                                             valid_bows)
    tfidf_similarities = cosine_similarity(valid_tfidf_mat, train_tfidf_mat)
    # Combine Graph2Vector and TF.IDF similarities
    similarity_matrix = _calc_similarities(g2v_similarities, \
                                           tfidf_similarities, alpha)
    # Find k nearest neighbors
    nn_matrix = _calc_nearest_neighbor_mat(similarity_matrix)
    knn_matrix = _find_k_nearest_neighbors(k, nn_matrix)
    predicted_grades = _predict_grades(knn_matrix, similarity_matrix, \
                                       valid_erd_map, train_erd_map, dataset_no, grades_dict)
    return _root_mean_square_error(predicted_grades, valid_erd_map, \
                                   dataset_no, grades_dict)


def _evaluate_over_splits(dataset: list, dataset_no: int, partitions: list, \
                          alpha: float, k: int, grades_dict: dict) -> float:
    """Given the dataset, dataset number, list of partition boundaries, the
    hyperparameters alpha and k, and the grades dictionary, evaluate the model
    over all splits.

    Returns the average RMSE of all the splits.
    """
    rmse_sum = 0.0
    for valid_range in partitions:
        error = _evaluate(dataset, dataset_no, valid_range, alpha, k, \
                          grades_dict)
        rmse_sum += error
    return rmse_sum / _NUM_PARTITIONS


def _evaluate_over_datasets(dataset1: list, dataset2: list, alpha: float, \
                            k: int, grades_dict: dict) -> float:
    """Given the two datasets, the hyperparameters alpha and k, and the grades
    dictionary, evaluate the model over the two datasets.

    Returns the average RMSE of all the datasets.
    """
    rmse_sum = 0.0
    random.shuffle(dataset1)
    dataset_no = 1
    partitions1 = _create_partitions(dataset1)
    rmse_sum += _evaluate_over_splits(dataset1, dataset_no, partitions1, \
                                      alpha, k, grades_dict)
    random.shuffle(dataset2)
    dataset_no = 2
    partitions2 = _create_partitions(dataset2)
    rmse_sum += _evaluate_over_splits(dataset2, dataset_no, partitions2, \
                                      alpha, k, grades_dict)
    return 0.5 * rmse_sum


def _tune_hyperparameters(dataset1: list, dataset2: list, grades_dict: dict) \
        -> tuple:
    """Given the two datasets and the grades dictionary, tune the
    hyperparameters alpha and k to minimize the average RMSE.

    Returns a tuple with the found ideal alpha and k, and the average RMSE.
    """
    max_k = int(max(len(dataset1), len(dataset2)) * (_NUM_PARTITIONS - 1) / \
                _NUM_PARTITIONS)
    k_step = int((max_k - _MIN_K) / (_NUM_K_STEPS - 1))
    if k_step < 1:
        k_step = 1
    optimal_alpha = -1.0
    optimal_k = -1
    min_rmse = math.inf
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X = []
    Y = []
    Z = []
    for i in range(_NUM_ALPHA_STEPS):
        alpha = _MIN_ALPHA + i * _ALPHA_STEP
        for k in range(_MIN_K, max_k - k_step, k_step):
            error = _evaluate_over_datasets(dataset1, dataset2, alpha, k, \
                                            grades_dict)
            if error < min_rmse:
                optimal_alpha = alpha
                optimal_k = k
                min_rmse = error
            X.append(alpha)
            Y.append(k)
            Z.append(error)
        k = max_k
        error = _evaluate_over_datasets(dataset1, dataset2, alpha, k, \
                                        grades_dict)
        if error < min_rmse:
            optimal_alpha = alpha
            optimal_k = k
            min_rmse = error
        X.append(alpha)
        Y.append(k)
        Z.append(error)
    ax.scatter(X, Y, Z)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('K')
    ax.set_zlabel('RMSE')
    plt.savefig('approach3_rmse.png')
    return optimal_alpha, optimal_k, min_rmse

#modified version based on the _evaluate function which can be used in line with the mainfile
def approach3(trainset, testset, traincollect, testcollect, testnames, gradepath, alpha=.02, k=39) -> list:
    #get the number of the dataset from one of the erd names
    dataset_no = int(trainset[0]['erd_id'].split('_')[1].split('.')[0])
    #create erd maps for both the train set and test set
    train_erd_map = _create_erd_map(trainset)
    test_erd_map = _create_erd_map(testset)

    #make the graph2vec and get similarities based on the training dataset and testing dataset
    train_g2v_matrix = ERDvec.make_graph2vec_matrix(trainset)
    test_g2v_matrix = ERDvec.make_graph2vec_matrix(testset)
    g2v_similarities = cosine_similarity(test_g2v_matrix, train_g2v_matrix)

    #create dictionary based on both collections (should be the same as creating dictionary on the whole dataset
    dictionary = Dictionary(traincollect+testcollect)
    tr_doc_bows = [dictionary.doc2bow(doc) for doc in traincollect]
    te_doc_bows  =[dictionary.doc2bow(doc) for doc in testcollect]

    #generate tfidf_models based on the training and testing bag of words
    tfidf_model = TfidfModel(tr_doc_bows)
    train_tfidf_mat = _generate_tfidf_matrix(tfidf_model, dictionary, \
                                             tr_doc_bows)
    valid_tfidf_mat = _generate_tfidf_matrix(tfidf_model, dictionary, \
                                             te_doc_bows)
    #compute tfidf similarities
    tfidf_similarities = cosine_similarity(valid_tfidf_mat, train_tfidf_mat)
    # Combine Graph2Vector and TF.IDF similarities
    similarity_matrix = _calc_similarities(g2v_similarities, \
                                           tfidf_similarities, alpha)

    #find nearest neighbors
    nn_matrix = _calc_nearest_neighbor_mat(similarity_matrix)
    knn_matrix = _find_k_nearest_neighbors(k, nn_matrix)
    #read grades in
    grades_dict = _read_grades(gradepath)

    #predict grades
    predicted_grades = _predict_grades(knn_matrix, similarity_matrix, \
                                       test_erd_map, train_erd_map, dataset_no, grades_dict)
    #restructure predictions to include file names
    results = []
    for i in range(len(predicted_grades)):
        results.append((testnames[i], predicted_grades[i]))
    return results


def main() -> None:
    """Main function."""
    _validate_args()
    dataset1_path = sys.argv[1]
    dataset1, itemcounts = util.clean_dataset(dataset1_path)
    dataset2_path = sys.argv[2]
    dataset2, itemcounts = util.clean_dataset(dataset2_path)
    del itemcounts
    grades_path = sys.argv[3]
    grades_dict = _read_grades(grades_path)
    # random.seed(0)
    start_time = time.perf_counter()
    alpha, k, min_rmse = _tune_hyperparameters(dataset1, dataset2, grades_dict)
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60.0
    elapsed_hours = elapsed_minutes / 60.0
    print('k = %d' % k)
    print('alpha = %.1f' % alpha)
    print('RMSE = %.4f' % min_rmse)
    print('Elapsed time = %.3fs (%.3fmin) (%.3fhrs)' % \
          (elapsed_seconds, elapsed_minutes, elapsed_hours))


if __name__ == '__main__':
    main()