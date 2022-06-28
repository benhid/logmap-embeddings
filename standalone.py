#!/usr/bin/env python
# coding: utf-8

import argparse
import configparser
import json
import random
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow_hub as hub
from gensim.models import Word2Vec
from nltk import word_tokenize
from numpy import linalg
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from extraction import get_class_names, get_classes

random.seed(42)


def encoder_vector(v: str, wv_model: Word2Vec) -> np.array:
    wv_dim = wv_model.vector_size
    if v in wv_model.wv:
        return wv_model.wv.get_vector(v, norm=True)
    else:
        print("encoder vector: ", v, "not in wv_model")
        return np.zeros(wv_dim)


def encoder_sentence(text: str, model) -> np.array:
    # The embeddings produced by the Universal Sentence Encoder are approximately normalized.
    return np.ravel(model([text]))


def encoder_words_avg(text: str, wv_model: Word2Vec) -> np.array:
    """
    Vector averaging means that the resulting vector is insensitive to the order of the words, i.e.,
    loses the word order in the same way as the standard bag-of-words models do.
    """
    wv_dim = wv_model.vector_size
    words = [token.lower() for token in word_tokenize(text) if token.isalpha()]
    # Remove out-of-vocabulary words.
    words = [word for word in words if word in wv_model.wv]
    vectors = [wv_model.wv.get_vector(word, norm=True) for word in words]
    if len(words) >= 1:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(wv_dim)


def cosine_distance(a, b):
    """
    Cosine distance is actually defined as `1 - cosine_similarity`.

    Since the cosine varies between -1 and 1, the result varies between 0 and 2.

    See:
    * https://stackoverflow.com/a/3767475/8025382
    """
    return 1 - spatial.distance.cosine(a, b)


def calc_distances(
    mappings,
    o2v_model: Word2Vec,
    wv_model: Word2Vec,
    *,
    use_sentences: bool = True,
    verbose: bool = False,
):
    assert o2v_model.vector_size == wv_model.vector_size

    num = len(mappings)

    y = np.zeros((num, 2))

    if use_sentences:
        # Load pre-trained universal sentence encoder model.
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    for i in tqdm(range(num), disable=not verbose):
        mapping = mappings[i].strip().split("|")

        c1, c2, l1, l2 = mapping[1:]

        if use_sentences:
            a = encoder_sentence(text=l1, model=embed)
            b = encoder_sentence(text=l2, model=embed)
        else:
            a = encoder_words_avg(text=l1, wv_model=wv_model)
            b = encoder_words_avg(text=l2, wv_model=wv_model)
        y[i, 0] = cosine_distance(a.tolist(), b.tolist())

        a = encoder_vector(v=c1, wv_model=o2v_model)
        b = encoder_vector(v=c2, wv_model=o2v_model)
        y[i, 1] = cosine_distance(a.tolist(), b.tolist())

    return y


if __name__ == "__main__":
    """
    Unsupervised distance-based entity alignment.

    Usage bioml track:
        $ mkdir cache_standalone-dist_bioml
        $ python standalone-dist.py --cache-dir cache_standalone-dist_bioml --config default_bioml.cfg > cache_standalone-dist_bioml/log_standalone.txt ; curl -X POST https://ghostie.link/standalone-ernesto

    Usage anatomy track:
        $ mkdir cache_standalone-dist_anatomy
        $ python standalone-dist.py --cache-dir cache_standalone-dist_anatomy --config default_anatomy.cfg > cache_standalone-dist_anatomy/log_standalone.txt
    """

    # Initiate the parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
    )
    parser.add_argument("--config", type=str, default="default.cfg")
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=False
    )

    # Read arguments from the command line.
    args = parser.parse_args()

    # Read configuration from file.
    config = configparser.ConfigParser()
    config.read(args.config)

    # All intermediary files go in the cache folder.
    cache_dir = args.cache_dir
    cache_dir.mkdir(exist_ok=True, parents=True)

    print("Found cache dir:", cache_dir.absolute())

    cached_names = cache_dir / "names.json"

    if not cached_names.is_file():
        print("Reading names")

        start = time.time()

        classes = get_classes(Path(config["default"]["ontology"]))
        names = {}

        for cls in classes:
            names[cls.iri] = get_class_names(cls=cls)

        print("took", timedelta(seconds=time.time() - start), "seconds")

        with open(cache_dir / "names.json", "w") as f:
            json.dump(names, f)
    else:
        print("Found names:", cached_names.absolute())

        with open(cached_names, "r") as f:
            names = json.load(f)

    # Read set of high recall candidate mappings for evaluation.
    print("Reading mappings for evaluation")

    start = time.time()

    overestimation_mappings = []

    with open(
        Path(config["default"]["logmap_output_dir"]) / "logmap_overestimation.txt", "r"
    ) as infile:
        overestimation = infile.readlines()

    for i, line in tqdm(enumerate(overestimation), disable=not args.verbose):
        mapping = line.strip().split("|")

        c1, c2 = mapping[0:2]

        n1 = names.get(c1)
        n2 = names.get(c2)

        if n1 and n2:
            for l1, l2 in [(x, y) for x in n1 for y in n2]:
                m = f"{i + 1}|{c1}|{c2}|{l1}|{l2}"
                overestimation_mappings.append(m)

    print("Total overestimation mappings =", len(overestimation_mappings))

    print("took", timedelta(seconds=time.time() - start), "seconds")

    with open(cache_dir / "samples.txt", "w") as f:
        for mapping in overestimation_mappings:
            f.write(mapping + "\n")

    # TODO - Load required models based on encoder_type.
    print("Loading pretrained models")

    start = time.time()

    ov_model = Word2Vec.load(config["model"]["owl2vec"])
    wv_model = Word2Vec.load(config["model"]["word2vec"])

    print("took", timedelta(seconds=time.time() - start), "seconds")

    print("Computing distances")

    start = time.time()

    y = calc_distances(
        mappings=overestimation_mappings,
        o2v_model=ov_model,
        wv_model=wv_model,
        use_sentences=True,
        verbose=args.verbose,
    )

    print("took", timedelta(seconds=time.time() - start), "seconds")

    with open(cache_dir / "distances.txt", "w") as f:
        for i, mapping in enumerate(overestimation_mappings):
            f.write("%s|%s\n" % (mapping.strip(), "|".join(["%.3f" % x for x in y[i]])))

    """
    X1, X2 = load_samples(
        mappings=overestimation_mappings,
        o2v_model=ov_model,
        wv_model=wv_model,
        encoder_type=encoder_type,
    )
    
    print("took", timedelta(seconds=time.time() - start), "seconds")

    assert X1.shape == X2.shape
    
    y_pred = [cosine_distance_scipy(X1[i].tolist(), X2[i].tolist()) for i in range(len(X1))]

    with open(cache_dir / "prediction.txt", "w") as f:
        for i, mapping in enumerate(overestimation_mappings):
            f.write("%s|%.3f\n" % (mapping.strip(), y_pred[i]))
    """
