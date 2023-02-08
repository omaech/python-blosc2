.. _NDArray:

NDArray API
===============

The multidimensional data array class. This class consists of a set of useful parameters and methods that allow not only to define an array correctly, but also to handle it in a simple way, being able to extract multidimensional slices from it.

.. currentmodule:: blosc2.NDArray

Attributes
----------

.. autosummary::
    :toctree: autofiles/ndarray

    ndim
    shape
    ext_shape
    chunks
    ext_chunks
    blocks
    chunksize
    size

Methods
-------

.. autosummary::
    :toctree: autofiles/ndarray
    :nosignatures:

    __getitem__
    __setitem__
    copy
    slice
    squeeze
    resize
    to_buffer

.. currentmodule:: blosc2

Constructors
------------

.. autosummary::
    :toctree: autofiles/ndarray
    :nosignatures:

    asarray
    copy
    empty
    from_buffer
    full
    zeros