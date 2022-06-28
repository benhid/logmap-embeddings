import argparse
from pathlib import Path

import networkx as nx
from gensim.models.word2vec import Word2Vec
from karateclub.utils.walker import BiasedRandomWalker, RandomWalker
from owlready2 import entity, get_ontology


def superclasses_of(cls) -> list:
    supclasses = set()
    for supclass in cls.is_a:
        if type(supclass) == entity.ThingClass:
            supclasses.add(supclass)
    return list(supclasses)


if __name__ == "__main__":
    # Initiate the parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("onto", type=str, help="Path to the ontology file.")
    parser.add_argument("--walk-number", type=int, default=5)
    parser.add_argument("--walk-length", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deepwalk_model/"),
        help="Path to the output model.",
    )

    # Read arguments from the command line.
    args = parser.parse_args()

    # Read ontology.
    onto = get_ontology(args.onto).load()

    parents = []
    n_mappings = 0

    for cls in onto.classes():
        # Mappings.
        mappings = cls.equivalent_to
        for mapping in mappings:
            if type(mapping) == entity.ThingClass:
                n_mappings += 1
                parents.append([name_to_string(cls.name), name_to_string(mapping.name)])
        # Super classes.
        supclasses = superclasses_of(cls=cls)
        for supclass in supclasses:
            parents.append([name_to_string(cls.name), name_to_string(supclass.name)])

    print("Mappings =", n_mappings)

    print("Relationships =", len(parents))

    G = nx.Graph(parents)

    def word2vec_model(output_path: str) -> None:
        model = Word2Vec(
            walker.walks,
            hs=1,
            alpha=0.05,
            epochs=10,
            vector_size=200,
            window=5,
            min_count=1,
            workers=4,
            seed=42,
        )
        model.save(output_path)

    # Proximity Preserving Node Embeddings.

    print(
        f"Fitting a DeepWalk model w/ walk_number={args.walk_number} walk_length={args.walk_length}"
    )

    walker = RandomWalker(walk_number=args.walk_number, walk_length=args.walk_length)
    walker.do_walks(G)

    print(f"Finished after {len(walker.walks)} walks.")

    word2vec_model(str(args.output / "deepwalk_output"))
