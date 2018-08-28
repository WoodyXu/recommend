# -*- encoding: utf-8 -*-

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd
import numpy as np

import collections
import scipy.sparse

import argparse
import json

SMOOTHING = 20.0
K1 = 100.0
B = 0.5

parser = argparse.ArgumentParser(description="Recommender Similarities")
parser.add_argument("--overlap", action="store_true", help="use overlap similarity")
parser.add_argument("--jaccard", action="store_true", help="use jaccard similarity")
parser.add_argument("--cosine", action="store_true", help="use cosine similarity")
parser.add_argument("--smoothed-cosine", action="store_true", help="use smoothed_cosine similarity")
parser.add_argument("--tfidf", action="store_true", help="use tfidf similarity")
parser.add_argument("--bm25", action="store_true", help="use bm25 similarity")

args = parser.parse_args()

def norm2(artist):
    if hasattr(artist, "norm2"):
        return getattr(artist, "norm2")

    ret = np.sqrt((artist.data ** 2).sum())
    setattr(artist, "norm2", ret)
    return ret


def binarize(artist):
    if hasattr(artist, "binarize"):
        return getattr(artist, "binarize")

    ret = scipy.sparse.csr_matrix(artist)
    ret.data = np.ones(len(artist.data))
    setattr(artist, "binarize", ret)
    return ret


def tfidf_weight(artist, idf):
    if hasattr(artist, "tfidf_weight"):
        return getattr(artist, "tfidf_weight")

    ret = scipy.sparse.csr_matrix(artist)
    ret.data = np.array([np.sqrt(plays) * idf[userid] 
        for plays, userid in zip(artist.data, artist.indices)])
    setattr(artist, "tfidf_weight", ret)
    return ret


def bm25_weight(artist, idf, average_plays):
    if hasattr(artist, "bm25_weight"):
        return getattr(artist, "bm25_weight")

    ret = scipy.sparse.csr_matrix(artist)
    length_norm = (1.0 - B) + B * sum(artist.data) * 1.0 / average_plays
    ret.data = np.array([(plays * (K1 + 1.0) / (K1 * length_norm + plays)) * idf[userid] 
            for plays, userid in zip(ret.data, ret.indices)])
    setattr(artist, "bm25_weight", ret)
    return ret


def overlap(set_a, set_b):
    return len(set_a.intersection(set_b))


def jaccard(set_a, set_b):
    intersection = len(set_a.intersection(set_b))
    return intersection * 1.0 / (len(set_a) + len(set_b) - intersection)


def cosine(artist_a, artist_b):
    return artist_a.dot(artist_b.T).toarray()[0][0] * 1.0 / (norm2(artist_a) * norm2(artist_b))


def smoothed_cosine(artist_a, artist_b):
    overlap = binarize(artist_a).dot(binarize(artist_b).T).toarray()[0][0]
    return (overlap * 1.0 / (SMOOTHING + overlap)) * cosine(artist_a, artist_b)


def tfidf(artist_a, artist_b, idf):
    tfidf_a = tfidf_weight(artist_a, idf)
    tfidf_b = tfidf_weight(artist_b, idf)

    return cosine(tfidf_a, tfidf_b)


def bm25(artist_a, artist_b, idf, average_plays):
    bm25_a = bm25_weight(artist_a, idf, average_plays)
    bm25_b = bm25_weight(artist_b, idf, average_plays)

    return bm25_a.dot(bm25_b.T).toarray()[0][0]


def serialize_topk(similarities, top_k=10):
    sorted_sim = sorted(similarities, reverse=True)
    sorted_sim = sorted_sim[: top_k]
    topks = []
    for index, (score, candidate) in enumerate(sorted_sim):
        topks.append({
                "rank": index + 1,
                "score": score,
                "artist": candidate
                })
    return topks


data = pd.read_table("lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv",
        usecols=[0, 2, 3], names=["user", "artist", "plays"])

print data.shape
print len(data["user"].unique())
print len(data["artist"].unique())

artist_users_sets = dict((artist, set(users)) for artist, users in data.groupby("artist")["user"])
user_artists_sets = dict((user, set(artists)) for user, artists in data.groupby("user")["artist"])

userids = collections.defaultdict(lambda: len(userids))
data["userid"] = data["user"].map(userids.__getitem__)

artists = dict(
    (
    artist, 
    scipy.sparse.csr_matrix((np.array(group["plays"]), (np.zeros(len(group)), group["userid"])), 
        shape=[1, len(userids)])
    ) 
    for artist, group in data.groupby("artist")
)

num_artists = len(artists)
idf = [1.0 + np.log(num_artists * 1.0 / (1.0 + p)) for p in data.groupby("user").size()]
average_plays = data["plays"].sum() * 1.0 / num_artists

popular_artists = [(len(users), artist) for artist, users in artist_users_sets.iteritems()]
popular_artists.sort(reverse=True)
popular_artists = [artist for cnt, artist in popular_artists[:10]]
print "The Top 10 popular artists ares:", popular_artists

result = []
for artist in popular_artists:
    this_artist = {}
    this_artist["object"] = artist

    user_matrix = artists[artist]
    user_set = artist_users_sets[artist]

    candidate_artists = set()
    for user in user_set:
        candidate_artists.update(user_artists_sets[user])
    candidate_artists.remove(artist)
    candidate_artists = [candidate for candidate in candidate_artists if len(artist_users_sets.get(candidate, [])) > 0]

    # overlap
    if args.overlap:
        similarities = [(overlap(user_set, artist_users_sets[candidate]), candidate) for candidate in candidate_artists]
        this_artist["overlap"] = serialize_topk(similarities)

    # jaccard
    if args.jaccard:
        similarities = [(jaccard(user_set, artist_users_sets[candidate]), candidate) for candidate in candidate_artists]
        this_artist["jaccard"] = serialize_topk(similarities)

    # cosine
    if args.cosine:
        similarities = [(cosine(user_matrix, artists[candidate]), candidate) for candidate in candidate_artists]
        this_artist["cosine"] = serialize_topk(similarities)

    # smoothed cosine
    if args.smoothed_cosine:
        similarities = [(smoothed_cosine(user_matrix, artists[candidate]), candidate) for candidate in candidate_artists]
        this_artist["smoothed_cosine"] = serialize_topk(similarities)

    # tfidf
    if args.tfidf:
        similarities = [(tfidf(user_matrix, artists[candidate], idf), candidate) for candidate in candidate_artists]
        this_artist["tfidf"] = serialize_topk(similarities)

    # bm25
    if args.bm25:
        similarities = [(bm25(user_matrix, artists[candidate], idf, average_plays), candidate) for candidate in candidate_artists]
        this_artist["bm25"] = serialize_topk(similarities)

    result.append(this_artist)

f_w = open("./similarities.json", "w")
json.dump(result, f_w, indent=4)
f_w.close()
