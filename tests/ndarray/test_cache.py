#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np
import pytest

import blosc2

argnames = "shape, chunks, blocks, slices, dtype"
argvalues = [
    ([456], [258], [73], slice(0, 1), np.int32),
    ([77, 134, 13], [31, 13, 5], [7, 8, 3], (slice(3, 7), slice(50, 100), 7), np.float64),
    ([12, 13, 14, 15, 16], [5, 5, 5, 5, 5], [2, 2, 2, 2, 2], (slice(1, 3), ..., slice(3, 6)), np.float32),
]


@pytest.mark.parametrize(argnames, argvalues)
def test_ndarray(shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(nparray, chunks=chunks, blocks=blocks)
    b = blosc2.CacheSChunk(a)

    np_slice = a[slices]
    cache_slice = b[slices]
    assert cache_slice.shape == np_slice.shape
    np.testing.assert_almost_equal(cache_slice, np_slice)

    a_slice = a.slice(slices)
    cache_slice = b.eval(slices)
    assert cache_slice.shape == a.shape
    np.testing.assert_almost_equal(cache_slice[slices], a_slice[...])

    cache_arr = b.eval()
    np.testing.assert_almost_equal(cache_arr[...], a[...])
