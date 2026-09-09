import math
from artemis_api.services.vector_math import l2_normalize, truncate_and_normalize, cosine_similarity


def test_l2_normalize_unit_length():
    v = l2_normalize([3.0, 4.0])
    assert math.isclose(sum(x*x for x in v), 1.0, rel_tol=1e-8)


def test_truncate_then_renormalize():
    v = truncate_and_normalize([1.0, 1.0, 100.0], 2)
    assert len(v) == 2
    assert math.isclose(sum(x*x for x in v), 1.0, rel_tol=1e-8)


def test_cosine_for_normalized_vectors_is_dot_product():
    a = l2_normalize([1.0, 0.0])
    b = l2_normalize([1.0, 1.0])
    assert math.isclose(cosine_similarity(a, b), 2**-0.5, rel_tol=1e-8)
