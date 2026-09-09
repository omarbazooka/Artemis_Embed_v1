import math
from typing import Iterable, List


def l2_normalize(vector: Iterable[float]) -> List[float]:
    values = [float(x) for x in vector]
    norm = math.sqrt(sum(x * x for x in values))
    if norm <= 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return [x / norm for x in values]


def truncate_and_normalize(vector: Iterable[float], dimension: int) -> List[float]:
    values = [float(x) for x in vector]
    if len(values) < dimension:
        raise ValueError(f"embedding has {len(values)} values, cannot truncate to {dimension}")
    return l2_normalize(values[:dimension])


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    av = [float(x) for x in a]
    bv = [float(x) for x in b]
    if len(av) != len(bv):
        raise ValueError("vectors must have the same dimension")
    return float(sum(x * y for x, y in zip(av, bv)))
