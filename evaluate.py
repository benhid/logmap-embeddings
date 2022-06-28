import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from tqdm import tqdm


def read_rdf_mappings(file_path):
    tree = ET.parse(file_path)
    mappings_str = list()
    all_mappings_str = list()
    for t in tree.getroot():
        for m in t:
            if "map" in m.tag:
                for c in m:
                    mapping = list()
                    mv = "?"
                    for i, v in enumerate(c):
                        if i < 2:
                            for value in v.attrib.values():
                                mapping.append(value.lower())
                                break
                        if i == 3:
                            mv = v.text
                    all_mappings_str.append("|".join(mapping))
                    if not mv == "?":
                        mappings_str.append("|".join(mapping))
    return mappings_str, all_mappings_str


def calc_p_r_f1(ref_mappings, preds):
    recall_num = 0
    for ref_mapping in ref_mappings:
        if ref_mapping in preds:
            recall_num += 1

    R = recall_num / len(ref_mappings)

    precision_num = 0
    num = 0
    for prediction in preds:
        if prediction in ref_mappings:
            precision_num += 1
        num += 1

    P = precision_num / num

    F1 = 2 * P * R / (P + R)

    return R, P, F1


if __name__ == "__main__":
    """
    Usage bioml track:
        $ python evaluate-dist.py --cache-dir cache_standalone-dist_bioml/ --reference use_cases/oaei_22_bioml/reference.txt --distances cache_standalone-dist_bioml/distances.txt --mappings logmap_output_oaei_22_bioml/logmap2_mappings.txt > cache_standalone-dist_bioml/log_evaluate-dist.txt ; curl -X POST -d "evaluate-dist finished" https://ghostie.link/standalone-ernesto

    Usage anatomy track:
        $ python evaluate-dist.py --cache-dir cache_standalone-dist_anatomy/ --reference use_cases/oaei_21_anatomy/reference.txt --distances cache_standalone-dist_anatomy/distances.txt --mappings logmap_output_oaei_21_anatomy/logmap2_mappings.txt > cache_standalone-dist_anatomy/log_evaluate-dist.txt
    """

    # Initiate the parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("reference.txt"),
        help="Path to reference alignment file.",
    )
    parser.add_argument(
        "--mappings",
        type=Path,
        default=Path("logmap_output/logmap2_mappings.txt"),
        help="Path to the LogMap mappings file.",
    )
    parser.add_argument(
        "--distances",
        type=Path,
        default=Path("cache/distances.txt"),
        help="Path to the distances file.",
    )

    # Read arguments from the command line.
    args = parser.parse_args()

    # All intermediary files go in the cache folder.
    cache_dir = args.cache_dir

    # Gold standard.
    ref_mappings = []
    with open(args.reference, "r") as f:
        for line in tqdm(f.readlines()):
            tmp = line.strip().split("|")
            pair = (tmp[0].lower(), tmp[1].lower())
            ref_mappings.append("%s|%s" % pair)

    print("Total reference mappings =", len(ref_mappings))

    # Distances.
    with open(args.distances, "r") as f:
        dist = f.readlines()

    print("Total samples =", len(dist))

    # LogMap's mappings.
    mappings = []
    with open(args.mappings, "r") as f:
        for line in tqdm(f.readlines()):
            tmp = line.strip().split("|")
            pair = (tmp[0].lower(), tmp[1].lower())
            mappings.append("%s|%s" % pair)

    print("LogMap mappings =", len(mappings))

    # Baseline.
    R, P, F1 = calc_p_r_f1(ref_mappings, mappings)

    print("Precision=%.3f Recall=%.3f F1=%.3f preds=%d *" % (P, R, F1, len(mappings)))

    for t1, t2 in [
        (0.95, 0.95),
        (0.9, 0.95),
        (0.85, 0.9),
        (0.85, 0.85),
        (0.8, 0.85),
        (0.8, 0.8),
    ]:
        # We incorporate LogMap mappings in our predictions.
        preds = set(mappings)

        for line in dist:
            tmp = line.strip().split("|")
            pair = (tmp[1].lower(), tmp[2].lower())
            if float(tmp[-2]) == 1.00:
                preds.add("%s|%s" % pair)
            elif float(tmp[-2]) >= t1 and float(tmp[-1]) >= t2:
                preds.add("%s|%s" % pair)
            else:
                pass

        R, P, F1 = calc_p_r_f1(ref_mappings, preds)

        print(
            "Precision=%.3f Recall=%.3f F1=%.3f preds=%d t1=%s t2=%s"
            % (P, R, F1, len(preds), t1, t2)
        )
