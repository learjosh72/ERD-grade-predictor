"""erdvec.py

A representation of an ERD diagram graph as a vector. Entities and
relationships are considered to be nodes in said graph.
"""

from __future__ import annotations

import math
import numpy as np


class ERDvec:
    """ERDvec

    A class representing an ERD graph in vector form.

    Constructed from JSON data in dict form containing
    information of the ERD.

    Components are:
    - Components representing the number of each comment flag
    - One component for the number of identifying relationships
    - One component for the number of weak entities
    - One component for the number of strong entities with at least 1 primary
      key
    - One component for the total number of strong entities

    Similarity is measured via cosine.

    Provides a class method to create a map of cosine angles for a given
    dataset.
    """
    _FLAG_CODE_MAP = {
        'WEAK_ENTITY_NO_IDENT_REL': 0,
        'FK_DRAWN': 1,
        'RELATIONSHIP_DANGLING': 2,
        'ATTRIBUTE_UNCONNECTED': 3,
        'PK_NOT_DEFINED': 4,
        'CARDINALITY_GLYPH_CONFLICT': 5
    }

    _NUM_DIMENSIONS = len(_FLAG_CODE_MAP) + 4
    _TOLERANCE = 1e-7

    @classmethod
    def make_graph2vec_matrix(cls, dataset: list) -> np.ndarray:
        """Converts the data in the dataset into ERDvecs, then combines them
        into a matrix."""
        graph2vec = np.zeros((len(dataset), cls._NUM_DIMENSIONS))
        for i, erd_data in enumerate(dataset):
            vector = ERDvec(erd_data)
            graph2vec[i] = np.array(vector)
        return graph2vec

    def __init__(self, erd_data: dict):
        """Creates a new ERDVec from the given ERD data.

        erd_data is the JSON data in the form of a dict."""
        self._components = np.zeros(self._NUM_DIMENSIONS)
        self._name = erd_data['erd_id']
        flag_counts = self._count_flags(erd_data)
        i = 0
        for j in range(len(flag_counts)):
            self._components[i] = flag_counts[i]
            i += 1
        ident_rel_count = self._count_ident_rels(erd_data)
        self._components[i] = ident_rel_count
        i += 1
        weak_entities_count = self._count_weak_entities(erd_data)
        self._components[i] = weak_entities_count
        i += 1
        pk_strong_entity_count = self._count_pk_strong_entities(erd_data)
        self._components[i] = pk_strong_entity_count
        i += 1
        strong_entity_count = self._count_strong_entities(erd_data)
        self._components[i] = strong_entity_count

    def __getitem__(self, index: int) -> float:
        """Gets the value of the component at index."""
        return self._components[index]

    def __len__(self) -> int:
        """Gets the number of components of the vector."""
        return self._NUM_DIMENSIONS

    def __str__(self) -> str:
        """Produces a string representation of this vector."""
        return '%s: %s' % (self._name, str(self._components))

    def _count_flags(self, erd_data: dict) -> list:
        """Counts the number of each commment flag this ERD has received."""
        counts = [0.0] * len(self._FLAG_CODE_MAP)
        if 'comment' not in erd_data:
            return counts
        if 'flags' not in erd_data['comment']:
            return counts
        for flag in erd_data['comment']['flags']:
            code = flag['code']
            if code not in self._FLAG_CODE_MAP:
                continue
            index = self._FLAG_CODE_MAP[code]
            counts[index] += 1.0
        return counts

    def _count_ident_rels(self, erd_data: dict) -> float:
        """Counts the number of intentifying relationships."""
        count = 0.0
        for rel in erd_data['relationships']:
            if rel['kind'] == 'identifying_relationship':
                count += 1.0
        return count

    def _count_weak_entities(self, erd_data: dict) -> float:
        """Counts the number of weak entities."""
        count = 0.0
        for entity in erd_data['entities']:
            if entity['kind'] == 'weak_entity':
                count += 1.0
        return count

    def _count_pk_strong_entities(self, erd_data: dict) -> float:
        """
        Counts the number of strong entities with at least one primary key.
        """
        count = 0.0
        for entity in erd_data['entities']:
            if entity['kind'] == 'entity' and len(entity['primary_keys']) > 0:
                count += 1.0
        return count

    def _count_strong_entities(self, erd_data: dict) -> float:
        """Counts the total number of strong entities."""
        count = 0.0
        for entity in erd_data['entities']:
            if entity['kind'] == 'entity':
                count += 1.0
        return count