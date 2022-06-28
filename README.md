# LogMap-ML Embeddings

[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--030--77385--4__23-blue)](https://openaccess.city.ac.uk/id/eprint/25810/1/ESWC2021_ontology_alignment_LogMap_ML.pdf)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Work in Progress!

Check out the parent project [LogMap matcher](https://github.com/ernestojimenezruiz/logmap-matcher/).

## Get started

### Setup 

```console
$ python3 -m venv .venv

$ source .venv/bin/activate

$ python -m pip install -r requirements.txt
```

### Usage

#### Pre-process: Run the original system

Run LogMap 4.0:

```console
$ mkdir logmap_output_oaei_22_bioml

$ java -jar logmap/logmap-matcher-4.0.jar MATCHER \
   file:$(pwd)/use_cases/oaei_22_bioml/fma.body.owl file:$(pwd)/use_cases/oaei_22_bioml/snomed.body.owl $(pwd)/logmap_output_oaei_22_bioml/ true
```

This leads to LogMap initial set of candidate mappings or _anchors_
(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_a">)
and
over-estimation class mappings
(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_o">).

#### Get Embedding Models

You can either download the word2vec embedding by gensim (the one trained with a corpus of Wikipedia articles from 2018-[download](https://drive.google.com/file/d/1rm9uJEKG25PJ79zxbZUWuaUroWeoWbFR/view?usp=sharing)) or use the ontology-tailored [OWL2Vec\*](https://github.com/KRR-Oxford/OWL2Vec-Star) embedding. The ontologies use one common embedding model.

```bash
$ python deepwalk.py use_cases/oaei_22_bioml/merged_with_mappings.owl --walk-number 10 --walk-length 2 --output deepwalk_model/
```

#### Prepare Dataset

Use the provided [standalone](standalone.py) script:

```bash
$ python standalone.py --cache-dir cache_standalone-dist_bioml --config default_bioml.cfg 
```

LogMap-ML will extract the class labels for each class in both ontologies and generate high-confidence train mappings
(_seed mappings_ <img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_s">)
for training.

It will also create a samples dataset from a set of high recall candidate mappings (LogMap’s over-estimation mappings <img src="https://render.githubusercontent.com/render/math?math=\mathcal{M}_o">) for evaluation.

#### Evaluate

Assuming that gold standards (complete ground truth mappings) are given, Precision and Recall can be directly calculated by:

```bash
$ python evaluate.py --cache-dir cache_standalone-dist_bioml/ --reference use_cases/oaei_22_bioml/reference.txt --distances cache_standalone-dist_bioml/distances.txt --mappings logmap_output_oaei_22_bioml/logmap2_mappings.txt
```

## Publications

* Jiaoyan Chen, Ernesto Jimenez-Ruiz, Ian Horrocks, Denvar Antonyrajah, Ali Hadian, Jaehun Lee. **Augmenting Ontology Alignment by Semantic Embedding and Distant Supervision**. European Semantic Web Conference, ESWC 2021. ([PDF](https://openaccess.city.ac.uk/id/eprint/25810/1/ESWC2021_ontology_alignment_LogMap_ML.pdf))

## License

This project is licensed under the terms of the MIT - see the [LICENSE](LICENSE) file for details.