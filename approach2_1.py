import random

from gensim import corpora

import utilities as u
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import re
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def process(text):
    """Preprocess text to tokens"""
    if not text:
        return []
    text = str(text)
    text = re.sub(r'#', 'number', text)
    text = re.sub(r'_no$|_No$|Num$|_num$| no$', 'number', text)
    text = re.sub('([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace('_', ' ').replace('-', ' ')
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # tokens = word_tokenize(text)
    return text

#

def erd_embedding(erd_id, erd, model):
    tags = [f"{erd_id}_entity_{i}" for i in range(len(erd["entities"]))] + \
           [f"{erd_id}_rel_{j}" for j in range(len(erd["relationships"]))]
    vectors = [model.dv[tag] for tag in tags]
    return sum(vectors) / len(vectors)

def combine_tags(erd_id, erd):
    tags = [f"{erd_id}_entity_{i}" for i in range(len(erd["entities"]))] + \
           [f"{erd_id}_rel_{j}" for j in range(len(erd["relationships"]))]
    return tags


def map_cardinality(cardinality):

    if cardinality.lower() == "unc" or cardinality is None or cardinality == "":
        return "unclear"
    out = []
    for i in cardinality:
        if i == '0':
            out.append("zero")
        elif i == '1':
            out.append("one")
        elif i == 'm' or i == "M":
            out.append("many")
        else:
            return "unclear"

    return out[0] + "-to-" + out[1]



def normalize_data(dataset):
    collection = build_collection(dataset)
    collect = []
    for doc in collection:
        words = []
        for word in doc:
            # print(f"Before: {word}")
            word = u.normalize_word(word)
            # print(f"After: {word}")
            words.append(word)
        collect.append(words)
    #print(collection)

    dictionary = corpora.Dictionary(collect)
    corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in collect]
    #for i,c in dictionary.cfs.items():
    #    print(dictionary[i],c)

    wordfreqs = sorted([(dictionary[i],c) for i,c in dictionary.cfs.items()], key=lambda x:x[1], reverse=True)
    #print(wordfreqs)

    collect2 = []
    for doc in collection:
        words2 = []
        for word in doc:
            word = u.normalize_word2(word, wordfreqs)
            #REMOVE HERE IF USING OR NOT (COLLECTIO VERSION)
            #word2 = removestopwords(word)
            #word3 = stemming(word2)
            words2.append(word)
        collect2.append(words2)

    for doc in dataset:
        for entity in doc['entities']:
            # print(entity)
            for feature in entity.keys():
                # print(feature)
                if isinstance(entity[feature], list):
                    entity[feature] = [process(u.datawordnormal(x, wordfreqs)) for x in entity[feature]]
                else:
                    entity[feature] = u.datawordnormal(entity[feature], wordfreqs)
        for relationship in doc['relationships']:
            # print(relationship)
            for feature in relationship.keys():
                # if feature == "attributes":
                # print(type(relationship[feature]))
                if isinstance(relationship[feature], list):
                    # print(feature)
                    # print(type(relationship[feature]))
                    new_items = []
                    for i, item in enumerate(relationship[feature]):
                        # print(type(item))
                        new_item = {}
                        if isinstance(item, dict):
                            # print(type(item))
                            for key, value in item.items():
                                # print(value)
                                if key == "cardinality":
                                    # print(value, type(value))
                                    relationship[feature][i][key] = map_cardinality(value)
                                else:
                                    relationship[feature][i][key] = process(u.datawordnormal(value, wordfreqs))
                                # print(new_value)
                                # new_item[key] = new_value

                        else:
                            # print(feature)
                            relationship[feature][i] = u.datawordnormal(item, wordfreqs)
                        new_items.append(new_item)
                        # print(new_items)

                        if len(new_items) == 0:
                            feature = new_items
                else:
                    # print(feature)
                    relationship[feature] = u.datawordnormal(relationship[feature], wordfreqs)
    return dataset, collect2

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
                        docwords.append(x)
                else:
                    #print(e[k])
                    docwords.append(e[k])
        for r in doc['relationships']:
            for k in r.keys():
                if isinstance(r[k], list):
                    for x in r[k]:
                        if isinstance(x, dict):
                            docwords.append(x['name'])
                        else:
                            docwords.append(x)
                else:
                    docwords.append(r[k])
        collectionwords.append(docwords)


    return collectionwords
def erd_to_tokens(erd_data):
        """
        Convert ERD JSON to a sequence of tokens for Doc2Vec.
        Uses special prefixes to help the model distinguish different components.
        """
        tokens = []
        # print(erd_data)
        # 1. Process entities
        for entity in erd_data.get('entities', []):
            # Entity name with prefix
            if u.replaceword(entity['kind']) == 'entity':
                tokens.append(f"ENTITY {entity['name']}")
            else:
                tokens.append(f"WEAK_ENTITY {entity['name']}")

            # Primary keys
            for pk in entity.get('primary_keys', []):
                tokens.append(f"PK {pk}")

            # Attributes
            for attr in entity.get('attributes', []):
                tokens.append(f"ATTR {attr}")

        # 2. Process relationships
        for rel in erd_data.get('relationships', []):
            # Relationship name
            if u.replaceword(rel["kind"]) == "relationship":
                tokens.append(f"REL {rel['name']}")
            else:
                tokens.append(f"WEAK_REL {rel['name']}")

            # Relationship attributes
            for attr in rel.get('attributes', []):
                tokens.append(f"RELATTR {attr}")

            # Process involved entities with cardinality
            for involved in rel.get('involved_entities', []):
                entity_name = involved['name']
                cardinality = involved['cardinality']

                # Create tokens that capture the relationship structure
                tokens.append(f"INVOLVES {entity_name}")
                tokens.append(f"CARD {cardinality}")

                # Create composite token for entity-relationship-cardinality pattern
                # tokens.append(f"{rel['name']}_{entity_name} {cardinality}")

        # 3. Optional: Add structural features
        # tokens.append(f"NUM_ENTITIES_{len(erd_data.get('entities', []))}")
        # tokens.append(f"NUM_RELATIONSHIPS_{len(erd_data.get('relationships', []))}")

        return tokens


def prepare_corpus(erd_documents):
        """
        Prepare a corpus of TaggedDocuments for Doc2Vec training.

        Args:
            erd_documents: List of ERD JSON objects

        Returns:
            List of TaggedDocument objects
        """
        tagged_docs = []

        for idx, erd_data in enumerate(erd_documents):
            # print(erd_data)
            tokens = erd_to_tokens(erd_data)
            # print(tokens)
            erd_id = erd_data.get('erd_id', f'doc_{idx}')

            # Create TaggedDocument with tokens and unique tag
            tagged_docs.append(TaggedDocument(words=tokens, tags=[erd_id]))

        return tagged_docs


    # Example usage
# def train_doc2vec_model(erd_documents, vector_size=100, epochs=40):
#         """
#         Train a Doc2Vec model on ERD documents.
#         """
#         # Prepare corpus
#         # print(erd_documents)
#         corpus = prepare_corpus(erd_documents)
#
#         # Initialize and train model
#         model = Doc2Vec(
#             vector_size=vector_size,
#             min_count=1,  # Keep all tokens since ERDs might have unique attributes
#             epochs=epochs,
#             dm=1,  # Use distributed memory (PV-DM)
#             window=5,
#             workers=4
#         )
#
#         # Build vocabulary
#         model.build_vocab(corpus)
#
#         # Train model
#         model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
#
#         return model


def normalize_similarity(cosine_sim):
    """
    Convert cosine similarity to [0, 1] range.
    Cosine similarity ranges from [-1, 1], we map it to [0, 1].
    """
    return (cosine_sim + 1) / 2


def prepare_training_data(doc2vec_model, training_data):
    """
    Prepare training data structures.

    Args:
        doc2vec_model: Trained Doc2Vec model
        training_data: List of dicts with 'erd_id' and 'grade' keys
                      e.g., [{'erd_id': '1_1.png', 'grade': 85.5}, ...]

    Returns:
        Tuple of (training_ids, training_embeddings, training_grades, global_mean)
    """
    training_ids = [item['erd_id'] for item in training_data]
    training_grades = {item['erd_id']: item['grade'] for item in training_data}

    # Get embeddings for all training ERDs
    embeddings = []
    for erd_id in training_ids:
        embeddings.append(doc2vec_model.dv[erd_id])
    training_embeddings = np.array(embeddings)

    # Calculate global mean for fallback
    grades = [item['grade'] for item in training_data]
    # print(grades)
    global_mean = np.mean(grades)

    # print(f"Prepared {len(training_ids)} training samples")
    # print(f"Global mean grade: {global_mean:.2f}")

    return training_ids, training_embeddings,training_grades


def knn_predict_single(query_embedding, training_ids, training_embeddings,
                       training_grades, global_mean=0, k=3,
                       similarity_threshold=0.3, return_details=False):
    """
    Predict grade for a single ERD using KNN regression.

    Args:
        query_embedding: Embedding vector for the query ERD
        training_ids: List of training ERD IDs
        training_embeddings: numpy array of training embeddings
        training_grades: Dict mapping erd_id to grade
        global_mean: Global mean grade for fallback
        k: Number of nearest neighbors to consider
        similarity_threshold: Minimum similarity threshold q
        return_details: If True, return detailed info about neighbors

    Returns:
        predicted_grade (float) or (predicted_grade, details) if return_details=True
    """
    # Ensure query embedding is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Compute cosine similarities with all training samples
    cosine_sims = cosine_similarity(query_embedding, training_embeddings)[0]

    # Normalize to [0, 1]
    similarities = normalize_similarity(cosine_sims)

    # Get top K similar ERDs (sorted by similarity)
    top_k_indices = np.argsort(similarities)[::-1][:k]

    # Filter by threshold
    valid_neighbors = []
    for idx in top_k_indices:
        sim = similarities[idx]
        if sim >= similarity_threshold:
            erd_id_neighbor = training_ids[idx]
            grade = training_grades[erd_id_neighbor]
            valid_neighbors.append({
                'erd_id': erd_id_neighbor,
                'similarity': sim,
                'grade': grade
            })

    # Calculate prediction
    if len(valid_neighbors) == 0:
        # Fallback to global mean
        prediction = global_mean
        method = 'global_mean'
    else:
        # Weighted average using similarities as weights
        total_weight = sum(n['similarity'] for n in valid_neighbors)
        weighted_sum = sum(n['similarity'] * n['grade'] for n in valid_neighbors)
        prediction = weighted_sum / total_weight
        method = 'weighted_knn'

    if return_details:
        details = {
            'prediction': prediction,
            'method': method,
            'k_used': len(valid_neighbors),
            'neighbors': valid_neighbors,
            'global_mean': global_mean
        }
        return prediction, details

    return prediction


def knn_predict_batch(doc2vec_model, test_erd_ids, training_ids,
                      training_embeddings, training_grades, global_mean,
                      k=3, similarity_threshold=0.3, queries=None):
    """
    Predict grades for multiple ERDs.

    Args:
        doc2vec_model: Trained Doc2Vec model
        test_erd_ids: List of ERD IDs to predict
        training_ids: List of training ERD IDs
        training_embeddings: numpy array of training embeddings
        training_grades: Dict mapping erd_id to grade
        global_mean: Global mean grade for fallback
        k: Number of nearest neighbors
        similarity_threshold: Minimum similarity threshold q

    Returns:
        List of predictions
    """
    predictions = []
    if queries is None:
        for erd_id in test_erd_ids:
            query_embedding = doc2vec_model.dv[erd_id]
            pred = knn_predict_single(
                query_embedding, training_ids, training_embeddings,
                training_grades, global_mean, k, similarity_threshold
            )
            predictions.append((erd_id, pred))
    else:
        for query, id in zip(queries, test_erd_ids):
            pred = knn_predict_single(np.array(query), training_ids, training_embeddings,
                                      training_grades, global_mean, k, similarity_threshold
                                      )
            predictions.append((id, pred))
    return predictions




def evaluate_predictions(doc2vec_model, test_data, training_ids,
                         training_embeddings, training_grades, global_mean,
                         k=3, similarity_threshold=0.3):
    """
    Evaluate predictions on test data.

    Args:
        doc2vec_model: Trained Doc2Vec model
        test_data: List of dicts with 'erd_id' and 'grade' keys
        training_ids: List of training ERD IDs
        training_embeddings: numpy array of training embeddings
        training_grades: Dict mapping erd_id to grade
        global_mean: Global mean grade for fallback
        k: Number of nearest neighbors
        similarity_threshold: Minimum similarity threshold q

    Returns:
        Dictionary with evaluation metrics
    """
    test_ids = [item['erd_id'] for item in test_data]
    true_grades = [item['grade'] for item in test_data]

    predictions_with_ids = knn_predict_batch(
        doc2vec_model, test_ids, training_ids, training_embeddings,
        training_grades, global_mean, k, similarity_threshold
    )
    ids = [x[0] for x in predictions_with_ids]
    predictions = [x[1] for x in predictions_with_ids]

    # Calculate metrics
    # print(test_data)
    errors = np.array(true_grades) - np.array(predictions)
    # print(errors)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    return {
        'mae': mae,
        'rmse': rmse,
        'predictions': predictions_with_ids,
        'true_grades': true_grades
    }

def splits(dataset, grades, testing = True, seed = 87):
    trainingdata = []
    testingdata = []
    trainingdatanames = []
    testingdatanames = []

    # print(len(grades))
    # print(grades)
    nums = []
    if testing is True:
        random.seed(seed)
        nums = random.sample([d for d in grades.keys()], 70)
        # print(nums)
        for i,doc in enumerate(dataset):
            parts = doc['erd_id'].split("_")

            if parts[0] in nums:
                grade = float(grades[parts[0]])
                # print(grade)
                if grade is not None:
                    trainingdata.append({"erd_id": doc['erd_id'], "grade": grade})

                    trainingdatanames.append(parts[0])
                pts = doc['erd_id'].split(".")

        othernums = [d for d in grades.keys() if d not in nums]
        # print(othernums)
        for i,doc in enumerate(dataset):
            parts = doc['erd_id'].split("_")
            if parts[0] in othernums:

                grade = float(grades[parts[0]])

                if grade is not None:
                    # print("here")
                    testingdata.append({"erd_id": doc['erd_id'], "grade": grade})


                    testingdatanames.append(parts[0])

                pts = doc['erd_id'].split(".")





    return trainingdata, testingdata, trainingdatanames, testingdatanames

def approach2(trainingdata, testingdata, normalized_data):
    print(trainingdata)
    corpus = prepare_corpus(normalized_data)
    model = Doc2Vec(vector_size=70, min_count=1, epochs=40, dm=1, window=7, workers=4)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

    training_ids, training_embeddings, training_grades = prepare_training_data(model, trainingdata)

    results = evaluate_predictions(
        model, testingdata, training_ids, training_embeddings,
        training_grades, global_mean=0, k=5, similarity_threshold=0.2
    )

    # print(f"RMSE: {results['rmse']:.2f}")
    #
    # print(results)
    predictions = results['predictions']
    true_grades = results['true_grades']
    # for pred, real in zip(predictions, true_grades):
    #     print(f"Predicted: {pred}, Real: {real}")
    return results

def run(gradepath, path="C:/Users/learj/PycharmProjects/pythonProject/cs473/project/for_students/for_students/Dataset1", seed=87, testing=None):
    dataset, itemcount = u.clean_dataset(path)
    # print(type(data))
    # print(type(data[0]))
    # print(data[0])
    # print(type(data[0][0]))

    normalized_data, collection2 = normalize_data(dataset)



    test_d1_grades = {}
    test_d2_grades = {}
    with open(gradepath, "r") as f:
        for line in f.readlines():
            parts = line.split(",")
            if parts[0] != "ERD_No":
                test_d1_grades[parts[0]] = parts[1]
                test_d2_grades[parts[0]] = parts[2]

    # print(test_d1_grades)
    #seed = 87
    trainingdata, testingdata, trainingdatanames, testingdatanames = splits(normalized_data, test_d1_grades, seed=seed)
    # print(testingdata)

    # print(trainingdata)
    corpus = prepare_corpus(normalized_data)
    model = Doc2Vec(vector_size=90, min_count=2, epochs=70, dm=1, window=4, workers=4)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

    training_ids, training_embeddings, training_grades = prepare_training_data(model, trainingdata)

    if testing is None:
        results = evaluate_predictions(
            model, testingdata, training_ids, training_embeddings,
            training_grades, global_mean=0, k=3, similarity_threshold=0.2
        )
        #print(results['rmse'])
        ps = []
        for p in results['predictions']:
            pparts = p[0].split("_")
            ps.append((pparts[0], p[1]))
        return ps
        #return results["predictions"]
    else:
        testing_set, testing_itemcount = u.clean_dataset(testing)
        # print(type(data))
        # print(type(data[0]))
        # print(data[0])
        # print(type(data[0][0]))

        normalized_testing, testing_collection2 = normalize_data(testing_set)
        # print(normalized_testing)
        testing_tokenized = [erd_to_tokens(erd) for erd in normalized_testing]
        erd_ids = [x['erd_id'] for x in normalized_testing]

        testing_embeddings = [model.infer_vector(x) for x in testing_tokenized]

        # print(erd_ids)

        # print(testing_tokenized)


        predictions = knn_predict_batch(model, erd_ids, training_ids,
                      training_embeddings, training_grades, global_mean=0,
                      k=3, similarity_threshold=0.4, queries=testing_embeddings)
        # print(predictions)
        ps = []
        for p in predictions:
            pparts = p[0].split("_")
            ps.append((pparts[0], p[1]))
        return ps
        #return predictions





        # predictions = knn_predict_batch(model, test_ids, training_ids, training_embeddings,
        # training_grades, global_mean=0, k=3, similarity_threshold=0.2)


        # results = approach2(trainingdata, testingdata, normalized_data)
        # print("Dataset1:")
        # print(f"RMSE: {results['rmse']:.2f}")
        # trainingdata2, testingdata2, trainingdatanames2, testingdatanames2 = splits(normalized_data, test_d2_grades,
        #                                                                             seed=seed)
        # print(testingdata)
        # results2 = approach2(trainingdata2, testingdata2, normalized_data)
        # print("Dataset2:")
        # print(f"RMSE: {results2['rmse']:.2f}")
        # print(results['predictions'])
        # return results["predictions"]

if __name__ == '__main__':

    run(testing="C:/Users/walte/PycharmProjects/CS473/Project/for_testing/Dataset1")
    # out = run(path="C:/Users/walte/PycharmProjects/CS473/Project/for_students/Dataset1")
    # print(out)
    # out2 = run(path="C:/Users/walte/PycharmProjects/CS473/Project/for_students/Dataset2")
    # print(out2)
    # dataset, itemcount = u.clean_dataset("C:/Users/walte/PycharmProjects/CS473/Project/for_students/Dataset1")
    # # print(type(data))
    # # print(type(data[0]))
    # # print(data[0])
    # # print(type(data[0][0]))
    #
    # normalized_data, collection2 = normalize_data(dataset)
    #
    # test_d1_grades = {}
    # test_d2_grades = {}
    # with open("C:/Users/walte/PycharmProjects/CS473/Project/for_students/ERD_grades.csv", "r") as f:
    #     for line in f.readlines():
    #         parts = line.split(",")
    #         if parts[0] != "ERD_No":
    #             test_d1_grades[parts[0]] = parts[1]
    #             test_d2_grades[parts[0]] = parts[2]
    #
    # # print(test_d1_grades)
    # seed = 87
    # trainingdata, testingdata, trainingdatanames, testingdatanames = splits(normalized_data, test_d1_grades, seed=seed)
    # # print(testingdata)
    # results = approach2(trainingdata, testingdata, normalized_data)
    # print("Dataset1:")
    # print(f"RMSE: {results['rmse']:.2f}")
    # trainingdata2, testingdata2, trainingdatanames2, testingdatanames2 = splits(normalized_data, test_d2_grades, seed=seed)
    # # print(testingdata)
    # results2 = approach2(trainingdata2, testingdata2, normalized_data)
    # print("Dataset2:")
    # print(f"RMSE: {results2['rmse']:.2f}")
    # for i in test_ERDs:
    #     with open("C:/Users/walte/PycharmProjects/CS473/Project/for_testing/Dataset1", "r")


    # Load your ERD data

    # Train model
    # model = train_doc2vec_model(normalized_data)

    # # Get embedding for a specific ERD
    # embedding = model.dv['1_1.png']  # Using erd_id as tag

    # Find similar ERDs
    # similar_erds = model.dv.most_similar('1_1.png', topn=5)
    # print("Most similar ERDs:", similar_erds)

    # Infer vector for new ERD
    # f = open("C:/Users/walte/PycharmProjects/CS473/Project/for_students/Dataset2/88_2.json", "r")
    # new_erd_document = json.load(f)
    # new_erd_data = new_erd_document  # Your new ERD JSON
    # new_tokens = erd_to_tokens(new_erd_data)
    # new_vector = model.infer_vector(new_tokens)