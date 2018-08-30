# -*- encoding: utf-8 -*-

import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import pandas as pd
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import argparse
import json

import implicit

parser = argparse.ArgumentParser(description="ALS")
parser.add_argument("--naive", action="store_true", help="use naive ALS")
parser.add_argument("--implicit", action="store_true", help="use implicit package ALS")

args = parser.parse_args()

def naive_implicit_als(sparse_data, alpha=40, iterations=10, lambda_value=0.1, factors=10):
    """
    Parameters:
        sparse_data: user-item csr-matrix
        alpha: the alpha value in function: confidence[u][i] = 1 + alpha * feedback[u][i]
        iterations: the fitting iterations of alternating least squares
        lambda_value: regularization factor
        factors: the latent factors for user latent matrix and item latent matrix
    Returns:
        X: user-factor csr-matrix
        Y: item-factor csr-matrix
    """
    # wait to plus 1
    confidence = alpha * sparse_data

    num_users, num_artists = sparse_data.shape

    X = scipy.sparse.csr_matrix(np.random.normal(size=(num_users, factors)))
    Y = scipy.sparse.csr_matrix(np.random.normal(size=(num_artists, factors)))

    X_identity = scipy.sparse.eye(num_users)
    Y_identity = scipy.sparse.eye(num_artists)
    factor_identity = lambda_value * scipy.sparse.eye(factors)

    for index in range(iterations):
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)

        # alternating computing
        for user_index in range(num_users):
            confidence_vec = confidence[user_index, :].toarray()
            preference_vec = confidence_vec.copy()
            preference_vec[preference_vec != 0] = 1.0
            confidence_diag = scipy.sparse.diags(confidence_vec, [0])

            A = yTy + Y.T.dot(confidence_diag).dot(Y) + factor_identity
            b = Y.T.dot(confidence_diag + Y_identity).dot(preference_vec.T)
            X[user_index] = scipy.sparse.linalg.spsolve(A, b)

        for artist_index in range(num_artists):
            confidence_vec = confidence[:, artist_index].T.toarray()
            preference_vec = confidence_vec.copy()
            preference_vec[preference_vec != 0] = 1.0
            confidence_diag = scipy.sparse.diags(confidence_vec, [0])

            A = xTx + X.T.dot(confidence_diag).dot(X) + factor_identity
            b = X.T.dot(confidence_diag + X_identity).dot(preference_vec.T)
            Y[artist_index] = scipy.sparse.linalg.spsolve(A, b)

    return X, Y

def nonzeros(mat, row):
    """
    Parameters:
        mat: csr_matrix, confidence
        row: row index
    Returns:
        iterator to represent non zero value column indices and data
    """
    for index in range(mat.indptr[row], mat.indptr[row+1]):
        yield mat.indices[index], mat.data[index]

def implicit_als(sparse_data, alpha=40, iterations=10, lambda_value=0.1, factors=10):
    """
    Parameters:
        sparse_data: user-item csr-matrix
        alpha: the alpha value in function: confidence[u][i] = 1 + alpha * feedback[u][i]
        iterations: the fitting iterations of alternating least squares
        lambda_value: regularization factor
        factors: the latent factors for user latent matrix and item latent matrix
    Returns:
        X: user-factor csr-matrix
        Y: item-factor csr-matrix
    """
    # wait to plus 1
    confidence = alpha * sparse_data
    num_users, num_artists = sparse_data.shape

    X = np.random.rand(num_users, factors) * 0.01
    Y = np.random.rand(num_artists, factors) * 0.01

    transpose_confidence = confidence.T.tocsr()

    for index in range(iterations):
        least_squares(confidence, X, Y, lambda_value)
        least_squares(transpose_confidence, Y, X, lambda_value)

    return X, Y

def least_squares(confidence, X, Y, lambda_value):
    num_users, num_factors = X.shape
    YtY = Y.T.dot(Y)

    for user_index in range(num_users):
        A = YtY + lambda_value * np.eye(num_factors)
        b = np.zeros(num_factors)

        for index, value in nonzeros(confidence, user_index):
            artist_factor = Y[index]
            A += value * np.outer(artist_factor, artist_factor)
            b += (value + 1) * artist_factor

        X[user_index] = np.linalg.solve(A, b)


data = pd.read_table("lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv",
    usecols=[0, 2, 3], names=["user", "artist", "plays"])
data = data.dropna()
data = data[data["plays"] != 0]

data["user_id"] = data["user"].astype("category").cat.codes
data["artist_id"] = data["artist"].astype("category").cat.codes

artist_id_name_d = {}
artist_name_id_d = {}
for index, row in data.iterrows():
    artist_id, artist_name = row["artist_id"], row["artist"]
    artist_id_name_d[str(artist_id)] = artist_name
    artist_name_id_d[artist_name] = str(artist_id)

users_set = list(np.sort(data["user_id"].unique()))
artists_set = list(np.sort(data["artist_id"].unique()))

sparse_user_item = scipy.sparse.csr_matrix(
    (data["plays"], (data["user_id"], data["artist_id"])),
    shape = (len(users_set), len(artists_set))
)

sparse_item_user = scipy.sparse.csr_matrix(
    (data["plays"], (data["artist_id"], data["user_id"])),
    shape = (len(artists_set), len(users_set))
)

popular_artists_id = [(len(users), artist_id) for artist_id, users in data.groupby("artist_id")["user_id"]]
popular_artists_id = sorted(popular_artists_id, reverse=True)[:10]
popular_artists_id = [artist[1] for artist in popular_artists_id]

if args.naive:
    user_hiddens, artist_hiddens = implicit_als(sparse_user_item, iterations=1)

if args.implicit:
    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)
    sparse_confidence = (15 * sparse_item_user).astype("double")
    model.fit(sparse_confidence)
    user_hiddens, artist_hiddens = model.user_factors, model.item_factors


sim_result = []
for artist_id in popular_artists_id:
    each_artist = {}
    each_artist["object"] = artist_id_name_d[str(artist_id)]
    each_artist["similarities"] = []

    artist_vec = artist_hiddens[artist_id]

    similarites = artist_hiddens.dot(artist_vec.reshape((-1, 1)))
    similarites = similarites.reshape(-1)
    sorted_idxes = np.argsort(similarites)[::-1][:10]
    for i, idx in enumerate(sorted_idxes):
        each_artist["similarites"].append(
            {
                "rank": i + 1,
                "artist": artist_id_name_d[str(idx)],
                "score": similarites[idx]
            }
        )
    sim_result.append(each_artist)

f_w = open("./similarities.json", "w")
json.dump(sim_result, f_w, indent=4)
f_w.close()
