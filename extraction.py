from pathlib import Path

from owlready2 import entity, get_ontology, owl

from lib.Label import label_to_string, name_to_string


def superclasses_of(cls: entity.ThingClass) -> list[entity.ThingClass]:
    supclasses = set()
    for supclass in cls.is_a:
        if type(supclass) == entity.ThingClass and supclass != owl.Thing:
            supclasses.add(supclass)
    return list(supclasses)


def get_class_path(cls: entity.ThingClass, p) -> list[str]:
    p.append(cls.iri)
    supclasses = superclasses_of(cls=cls)
    if owl.Thing in supclasses or len(supclasses) == 0:
        return p
    else:
        return get_class_path(cls=supclasses[0], p=p)


def get_class_names(cls: entity.ThingClass) -> list[str]:
    """
    Get the URI name and english labels of a class.
    """
    name = [name_to_string(cls.name)]
    labels = cls.label.en or cls.label
    labels = [label_to_string(label) for label in labels]
    return labels or name


def get_classes(onto: Path) -> list[entity.ThingClass]:
    onto = get_ontology(str(onto)).load()
    return onto.classes()
