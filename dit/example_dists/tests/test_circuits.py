"""
Tests for dit.example_dists.circuits.
"""

from nose.tools import assert_almost_equal

from dit.example_dists import (Unq, Rdn, Xor, RdnXor, ImperfectRdn, Subtle, And,
                               Or)

from dit.algorithms import insert_meet, pruned_samplespace
from dit.shannon import mutual_information

def test_unq():
    """ Test the Unq distribution """
    d = Unq()
    d = pruned_samplespace(d)
    d = insert_meet(d, -1, [[0], [1]])
    i1 = mutual_information(d, [0], [2])
    i2 = mutual_information(d, [1], [2])
    i12 = mutual_information(d, [0, 1], [2])
    r = mutual_information(d, [2], [3])
    assert_almost_equal(i1, 1)
    assert_almost_equal(i2, 1)
    assert_almost_equal(i12, 2)
    assert_almost_equal(r, 0)

def test_rdn():
    """ Test the Rdn distribution """
    d = Rdn()
    d = pruned_samplespace(d)
    d = insert_meet(d, -1, [[0], [1]])
    i1 = mutual_information(d, [0], [2])
    i2 = mutual_information(d, [1], [2])
    i12 = mutual_information(d, [0, 1], [2])
    r = mutual_information(d, [2], [3])
    assert_almost_equal(i1, 1)
    assert_almost_equal(i2, 1)
    assert_almost_equal(i12, 1)
    assert_almost_equal(r, 1)

def test_xor():
    """ Test the Xor distribution """
    d = Xor()
    d = pruned_samplespace(d)
    d = insert_meet(d, -1, [[0], [1]])
    i1 = mutual_information(d, [0], [2])
    i2 = mutual_information(d, [1], [2])
    i12 = mutual_information(d, [0, 1], [2])
    r = mutual_information(d, [2], [3])
    assert_almost_equal(i1, 0)
    assert_almost_equal(i2, 0)
    assert_almost_equal(i12, 1)
    assert_almost_equal(r, 0)

def test_and():
    """ Test the And distribution """
    d = And()
    d = pruned_samplespace(d)
    d = insert_meet(d, -1, [[0], [1]])
    i1 = mutual_information(d, [0], [2])
    i2 = mutual_information(d, [1], [2])
    i12 = mutual_information(d, [0, 1], [2])
    r = mutual_information(d, [2], [3])
    assert_almost_equal(i1, 0.31127812445913294)
    assert_almost_equal(i2, 0.31127812445913294)
    assert_almost_equal(i12, 0.81127812445913294)
    assert_almost_equal(r, 0)

def test_or():
    """ Test the Or distribution """
    d = Or()
    d = pruned_samplespace(d)
    d = insert_meet(d, -1, [[0], [1]])
    i1 = mutual_information(d, [0], [2])
    i2 = mutual_information(d, [1], [2])
    i12 = mutual_information(d, [0, 1], [2])
    r = mutual_information(d, [2], [3])
    assert_almost_equal(i1, 0.31127812445913294)
    assert_almost_equal(i2, 0.31127812445913294)
    assert_almost_equal(i12, 0.81127812445913294)
    assert_almost_equal(r, 0)

def test_rdnxor():
    """ Test the RdnXor distribution """
    d = RdnXor()
    d = pruned_samplespace(d)
    d = insert_meet(d, -1, [[0], [1]])
    i1 = mutual_information(d, [0], [2])
    i2 = mutual_information(d, [1], [2])
    i12 = mutual_information(d, [0, 1], [2])
    r = mutual_information(d, [2], [3])
    assert_almost_equal(i1, 1)
    assert_almost_equal(i2, 1)
    assert_almost_equal(i12, 2)
    assert_almost_equal(r, 1)

def test_imperfectrdn():
    """ Test the ImperfectRdn distribution """
    d = ImperfectRdn()
    d = pruned_samplespace(d)
    d = insert_meet(d, -1, [[0], [1]])
    i1 = mutual_information(d, [0], [2])
    i2 = mutual_information(d, [1], [2])
    i12 = mutual_information(d, [0, 1], [2])
    r = mutual_information(d, [2], [3])
    assert_almost_equal(i1, 1)
    assert_almost_equal(i2, 0.98959007894024409)
    assert_almost_equal(i12, 1)
    assert_almost_equal(r, 0)

def test_subtle():
    """ Test the Subtle distribution """
    d = Subtle()
    d = pruned_samplespace(d)
    d = insert_meet(d, -1, [[0], [1]])
    i1 = mutual_information(d, [0], [2])
    i2 = mutual_information(d, [1], [2])
    i12 = mutual_information(d, [0, 1], [2])
    r = mutual_information(d, [2], [3])
    assert_almost_equal(i1, 0.91829583405448956)
    assert_almost_equal(i2, 0.91829583405448956)
    assert_almost_equal(i12, 1.5849625007211561)
    assert_almost_equal(r, 0)
