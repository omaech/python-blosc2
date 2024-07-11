#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import pathlib

import numpy as np
import pytest

import blosc2

NITEMS_SMALL = 1_000
ROOT = "b2tests"
DIR = "expr/"


def get_array(shape, chunks_blocks):
    dtype = np.float64
    urlpath = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a1-{shape}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + urlpath}").as_posix()
    a1 = blosc2.C2Array(path)
    return a1


@pytest.mark.parametrize(
    "chunks_blocks",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
@pytest.mark.parametrize(
    "urlpath, slices",
    [
        (None, (slice(0, 23), slice(None))),
        ("proxy", (slice(None), slice(None))),
        (None, (slice(0, 5), slice(0, 60))),
        ("proxy", (slice(37, 53), slice(19, 233))),
    ],
)
def test_simple(chunks_blocks, c2sub_context, urlpath, slices):
    blosc2.remove_urlpath(urlpath)
    shape = (60, 60)
    a = get_array(shape, chunks_blocks)
    b = blosc2.ProxySChunk(a, urlpath=urlpath)

    np.testing.assert_allclose(b[slices], a[slices])

    cache_slice = b.eval(slices)
    assert cache_slice.schunk.urlpath == urlpath
    np.testing.assert_allclose(cache_slice[slices], a[slices])

    cache = b.eval()
    assert cache.schunk.urlpath == urlpath
    np.testing.assert_allclose(cache[...], a[...])

    blosc2.remove_urlpath(urlpath)


def test_small(c2sub_context):
    shape = (NITEMS_SMALL,)
    chunks_blocks = "default"
    a = get_array(shape, chunks_blocks)
    b = blosc2.ProxySChunk(a)

    np.testing.assert_allclose(b[0:100], a[0:100])

    cache_slice = b.eval(slice(0, 100))
    np.testing.assert_allclose(cache_slice[0:100], a[0:100])

    cache = b.eval()
    np.testing.assert_allclose(cache[...], a[...])
