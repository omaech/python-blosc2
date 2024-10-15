#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
# Avoid checking the name of type annotations at run time
from __future__ import annotations

import copy
import ctypes
import ctypes.util
import math
import os
import pathlib
import pickle
import platform
import sys
from dataclasses import asdict
from typing import TYPE_CHECKING

import cpuinfo
import numpy as np

import blosc2
from blosc2 import blosc2_ext

if TYPE_CHECKING:
    from collections.abc import Callable

    import tensorflow
    import torch


def _check_typesize(typesize):
    if not 1 <= typesize <= blosc2_ext.MAX_TYPESIZE:
        raise ValueError(f"typesize can only be in the 1-{blosc2_ext.MAX_TYPESIZE} range.")


def _check_clevel(clevel):
    if not 0 <= clevel <= 9:
        raise ValueError("clevel can only be in the 0-9 range.")


def _check_input_length(input_name, input_len, typesize, _ignore_multiple_size=False):
    if input_len > blosc2_ext.MAX_BUFFERSIZE:
        raise ValueError(f"{input_name} cannot be larger than {blosc2_ext.MAX_BUFFERSIZE} bytes")
    if not _ignore_multiple_size and input_len % typesize != 0:
        raise ValueError(f"len({input_name}) can only be a multiple of typesize ({typesize}).")


def _check_filter(filter):
    if filter not in blosc2.Filter:
        raise ValueError(f"filter can only be one of: {blosc2.Filter.keys()}")


def _check_codec(codec):
    if codec not in blosc2.Codec:
        raise ValueError(f"codec can only be one of: {codecs}, not '{codec}'")


def compress(
    src: object,
    typesize: int = 8,
    clevel: int = 1,
    filter: blosc2.Filter = blosc2.Filter.SHUFFLE,
    codec: blosc2.Codec = blosc2.Codec.ZSTD,
    _ignore_multiple_size: bool = False,
) -> str | bytes:
    """Compress src, with a given type size.

    Parameters
    ----------
    src: bytes-like object (supporting the buffer interface)
        The data to be compressed.
    typesize: int (optional) from 1 to 255
        The data type size. The default is 1, or `src.itemsize` if it exists.
    clevel: int (optional)
        The compression level from 0 (no compression) to 9
        (maximum compression).  The default is 9.
    filter: :class:`Filter` (optional)
        The filter to be activated. The
        default is :py:obj:`Filter.SHUFFLE <Filter>`.
    codec: :class:`Codec` (optional)
        The compressor used internally in Blosc. The default is :py:obj:`Codec.BLOSCLZ <Codec>`.

    Returns
    -------
    out: str or bytes
        The compressed data in form of a Python str or bytes object.

    Raises
    ------
    TypeError
        If :paramref:`src` doesn't support the buffer interface.
    ValueError
        If :paramref:`src` is too long.
        If :paramref:`typesize` is not within the allowed range.
        If :paramref:`clevel` is not within the allowed range.
        If :paramref:`codec` is not within the supported compressors.

    Notes
    -----
    The `cname` and `shuffle` parameters in python-blosc API have been replaced by :paramref:`codec` and
    :paramref:`filter` respectively.
    To set :paramref:`codec` and :paramref:`filter`, use the enumerations :class:`Codec` and :class:`Filter`
    instead of the python-blosc API variables like `blosc.SHUFFLE` for :paramref:`filter`
    or strings like "blosclz" for :paramref:`codec`.

    Examples
    --------
    >>> import array, sys
    >>> a = array.array('i', range(1000*1000))
    >>> a_bytesobj = a.tobytes()
    >>> c_bytesobj = blosc2.compress(a_bytesobj, typesize=4)
    >>> len(c_bytesobj) < len(a_bytesobj)
    True
    """
    len_src = len(src)
    if hasattr(src, "itemsize"):
        if typesize is None:
            typesize = src.itemsize
        len_src *= src.itemsize
    else:
        # Let's not guess the typesize for non NumPy objects
        if typesize is None:
            typesize = 1
    _check_clevel(clevel)
    _check_typesize(typesize)
    _check_filter(filter)
    _check_input_length("src", len_src, typesize, _ignore_multiple_size=_ignore_multiple_size)
    return blosc2_ext.compress(src, typesize, clevel, filter, codec)


def decompress(
    src: object, dst: object | bytearray = None, as_bytearray: bool = False
) -> str | bytes | bytearray | None:
    """Decompresses a bytes-like compressed object.

    Parameters
    ----------
    src: bytes-like object
        The data to be decompressed.  Must be a bytes-like object
        that supports the Python Buffer Protocol, like bytes, bytearray,
        memoryview, or
        `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_.
    dst: NumPy object or bytearray
        The destination NumPy object or bytearray to fill,
        the length of which must be greater than 0.
        The user must make sure
        that it has enough capacity for hosting the decompressed data.
        Default is None, meaning that a new `bytes` or `bytearray` object
        is created, filled and returned.
    as_bytearray: bool (optional)
        If this flag is True, then the return type will be a bytearray object
        instead of a bytes object.

    Returns
    -------
    out: str or bytes or bytearray
        If :paramref:`dst` is `None`, the decompressed data will be returned as a Python str or bytes object.
        If as_bytearray is True, the return type will be a bytearray object.

        If :paramref:`dst` is not `None`, the function will return `None` because the result
        will already be stored in :paramref:`dst`.

    Raises
    ------
    RuntimeError
        Raised if the compressed data is corrupted or the output buffer is not large enough.
        Also raised if a `bytes` object could not be obtained.
    TypeError
        Raised if :paramref:`src` does not support the Buffer Protocol.
    ValueError
        Raised if the length of :paramref:`src` is smaller than the minimum required length.
        Also raised if `dst` is not `None` and its length is 0.

    Examples
    --------
    >>> import array, sys
    >>> a = array.array('i', range(1000*1000))
    >>> a_bytesobj = a.tobytes()
    >>> c_bytesobj = blosc2.compress(a_bytesobj, typesize=4)
    >>> a_bytesobj2 = blosc2.decompress(c_bytesobj)
    >>> a_bytesobj == a_bytesobj2
    True
    >>> b"" == blosc2.decompress(blosc2.compress(b""))
    True
    >>> b"1"*7 == blosc2.decompress(blosc2.compress(b"1"*7))
    True
    >>> type(blosc2.decompress(blosc2.compress(b"1"*7),
    ...                        as_bytearray=True)) is bytearray
    True
    >>> import numpy as np
    >>> arr = np.arange(10)
    >>> comp_arr = blosc2.compress(arr)
    >>> dest = np.empty(arr.shape, arr.dtype)
    >>> blosc2.decompress(comp_arr, dst=dest)
    >>> np.array_equal(arr, dest)
    True
    """
    return blosc2_ext.decompress(src, dst, as_bytearray)


def pack(
    obj: object,
    clevel: int = 9,
    filter: blosc2.Filter = blosc2.Filter.SHUFFLE,
    codec: blosc2.Codec = blosc2.Codec.BLOSCLZ,
) -> str | bytes:
    """Pack (compress) a Python object.

    Parameters
    ----------
    obj: Python object  with `itemsize` attribute
        The Python object to be packed.
    clevel: int (optional)
        The compression level from 0 (no compression) to 9
        (maximum compression).  The default is 9.
    filter: :class:`Filter` (optional)
        The filter to be activated. The
        default is :py:obj:`Filter.SHUFFLE <Filter>`.
    codec: :class:`Codec` (optional)
        The compressor used internally in Blosc. The default is
        :py:obj:`Codec.BLOSCLZ <Codec>`.

    Returns
    -------
    out: str or bytes
        The packed object in form of a Python str or bytes object.

    Raises
    ------
    AttributeError
        If :paramref:`obj` does not have an `itemsize` attribute.
        If :paramref:`obj` does not have an `size` attribute.
    ValueError
        If the pickled object size is larger than the maximum allowed buffer size.
        If typesize is not within the allowed range.
        If :paramref:`clevel` is not within the allowed range.
        If :paramref:`codec` is not within the supported compressors.

    Notes
    -----
    The `cname` and `shuffle` parameters in python-blosc API have been replaced by :paramref:`codec` and
    :paramref:`filter` respectively.
    To set :paramref:`codec` and :paramref:`filter`, use the enumerations :class:`Codec` and :class:`Filter`
    instead of the python-blosc API variables such as `blosc.SHUFFLE` for :paramref:`filter`
    or strings like "blosclz" for :paramref:`codec`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(1e6)
    >>> parray = blosc2.pack(a)
    >>> len(parray) < a.size * a.itemsize
    True
    """
    if not hasattr(obj, "itemsize"):
        raise AttributeError("The object must have an itemsize attribute.")
    if not hasattr(obj, "size"):
        raise AttributeError("The object must have an size attribute.")

    itemsize = obj.itemsize
    _check_clevel(clevel)
    _check_codec(codec)
    _check_typesize(itemsize)
    pickled_object = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
    # The object to be compressed is pickled_object, and not obj
    len_src = len(pickled_object)
    _check_input_length("pickled object", len_src, itemsize, _ignore_multiple_size=True)
    return compress(
        pickled_object,
        typesize=itemsize,
        clevel=clevel,
        filter=filter,
        codec=codec,
        _ignore_multiple_size=True,
    )


def unpack(packed_object: str | bytes, **kwargs: dict) -> object:
    """Unpack (decompress) an object.

    Parameters
    ----------
    packed_object: str or bytes
        The packed object to be decompressed.
    kwargs: dict, optional
        Parameters that can be passed to the
        `pickle.loads API <https://docs.python.org/3/library/pickle.html#pickle.loads>`_

    Returns
    -------
    out: object
        The decompressed data in form of the original object.

    Raises
    ------
    TypeError
        If :paramref:`packed_object` is not of type bytes or string.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(1e6)
    >>> parray = blosc2.pack(a)
    >>> len(parray) < a.size * a.itemsize
    True
    >>> a2 = blosc2.unpack(parray)
    >>> np.array_equal(a, a2)
    True
    >>> a = np.array(['å', 'ç', 'ø'])
    >>> parray = blosc2.pack(a)
    >>> a2 = blosc2.unpack(parray)
    >>> np.array_equal(a, a2)
    True
    """
    pickled_object = decompress(packed_object)
    if kwargs:
        obj = pickle.loads(pickled_object, **kwargs)
    else:
        obj = pickle.loads(pickled_object)

    return obj


def pack_array(
    arr: np.ndarray,
    clevel: int = 9,
    filter: blosc2.Filter = blosc2.Filter.SHUFFLE,
    codec: blosc2.Codec = blosc2.Codec.BLOSCLZ,
) -> str | bytes:
    """Pack (compress) a NumPy array. It is equivalent to the pack function.

    Parameters
    ----------
    arr: np.ndarray
        The NumPy array to be packed.
    clevel: int (optional)
        The compression level from 0 (no compression) to 9
        (maximum compression).  The default is 9.
    filter: :class:`Filter` (optional)
        The filter to be applied during compression. The
        default is :py:obj:`Filter.SHUFFLE <Filter>`.
    codec: :class:`Codec` (optional)
        The codec to be used for compression. The default is
        :py:obj:`Codec.BLOSCLZ <Codec>`.

    Returns
    -------
    out: str or bytes
        The packed array in the form of a Python str or bytes object.

    Raises
    ------
    AttributeError
        If :paramref:`arr` does not have an `itemsize` attribute.
        If :paramref:`arr` does not have a `size` attribute.
    ValueError
        If typesize is not within the allowed range.
        If the pickled object size is larger than the maximum allowed buffer size.
        If :paramref:`clevel` is not within the allowed range.
        If :paramref:`codec` is not within the supported compressors.

    See also
    --------
    :func:`~blosc2.pack`

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(1e6)
    >>> parray = blosc2.pack_array(a)
    >>> len(parray) < a.size*a.itemsize
    True
    """
    return pack(arr, clevel, filter, codec)


def unpack_array(packed_array: str | bytes, **kwargs: dict) -> np.ndarray:
    """Restore a packed NumPy array.

    Parameters
    ----------
    packed_array: str or bytes
        The packed array to be restored.
    kwargs: dict, optional
        Parameters that can be passed to the
        `pickle.loads API <https://docs.python.org/3/library/pickle.html#pickle.loads>`_

    Returns
    -------
    out: ndarray
        The decompressed data in form of a NumPy array.

    Raises
    ------
    TypeError
        If :paramref:`packed_array` is not of type bytes or string.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(1e6)
    >>> parray = blosc2.pack_array(a)
    >>> len(parray) < a.size*a.itemsize
    True
    >>> a2 = blosc2.unpack_array(parray)
    >>> np.array_equal(a, a2)
    True
    >>> a = np.array(['å', 'ç', 'ø'])
    >>> parray = blosc2.pack_array(a)
    >>> a2 = blosc2.unpack_array(parray)
    >>> np.array_equal(a, a2)
    True
    """
    pickled_array = decompress(packed_array)
    if kwargs:
        arr = pickle.loads(pickled_array, **kwargs)
        if all(isinstance(x, bytes) for x in arr.tolist()):
            arr = np.array([x.decode("utf-8") for x in arr.tolist()])
    else:
        arr = pickle.loads(pickled_array)

    return arr


def pack_array2(arr: np.ndarray, chunksize: int = None, **kwargs: dict) -> bytes | int:
    """Pack (compress) a NumPy array. This is faster, and it does not have a 2 GB limitation.

    Parameters
    ----------
    arr: np.ndarray
        The NumPy array to be packed.

    chunksize: int
        The size (in bytes) for the chunks during compression. If not provided,
        it is computed automatically.

    Other parameters
    ----------------
    kwargs: dict, optional
        These are the same as the kwargs in :func:`SChunk.__init__ <blosc2.schunk.SChunk.__init__>`.

    Returns
    -------
    out: bytes | int
        The serialized version (cframe) of the array.
        If urlpath is provided, the number of bytes in file is returned instead.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(1e6)
    >>> cframe = blosc2.pack_array2(a)
    >>> len(cframe) < a.size * a.itemsize
    True

    See also
    --------
    :func:`~blosc2.unpack_array2`
    :func:`~blosc2.save_array`
    :func:`~blosc2.pack_tensor`
    :func:`~blosc2.save_tensor`
    """
    # May we raise a DeprecationWarning here in the future?
    return pack_tensor(arr, chunksize, **kwargs)


def unpack_array2(cframe: bytes) -> np.ndarray:
    """Unpack (decompress) a packed NumPy array via a cframe.

    Parameters
    ----------
    cframe: bytes
        The packed array to be restored.

    Returns
    -------
    out: np.ndarray
        The unpacked NumPy array.

    Raises
    ------
    TypeError
        If :paramref:`cframe` is not of type bytes, or not a cframe.
    RunTimeError
        If some other problem is detected.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(1e6)
    >>> cframe = blosc2.pack_array2(a)
    >>> len(cframe) < a.size*a.itemsize
    True
    >>> a2 = blosc2.unpack_array2(cframe)
    >>> np.array_equal(a, a2)
    True

    See also
    --------
    :func:`~blosc2.pack_array2`
    :func:`~blosc2.pack_tensor`
    :func:`~blosc2.save_array`
    :func:`~blosc2.save_tensor`
    """
    # May we raise a DeprecationWarning here in the future?
    return unpack_tensor(cframe)


def save_array(arr: np.ndarray, urlpath: str, chunksize: int = None, **kwargs: dict) -> int:
    """Save a serialized NumPy array in `urlpath`.

    Parameters
    ----------
    arr: np.ndarray
        The NumPy array to be saved.

    urlpath: str
        The path for the file where the array is saved.

    chunksize: int
        The size (in bytes) for the chunks during compression. If not provided,
        it is computed automatically.

    Other parameters
    ----------------
    kwargs: dict, optional
        These are the same as the kwargs in :func:`SChunk.__init__ <blosc2.schunk.SChunk.__init__>`.

    Returns
    -------
    out: int
        The number of bytes of the saved array.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(1e6)
    >>> serial_size = blosc2.save_array(a, "test.bl2", mode="w")
    >>> serial_size < a.size * a.itemsize
    True

    See also
    --------
    :func:`~blosc2.load_array`
    :func:`~blosc2.pack_array2`
    :func:`~blosc2.save_tensor`
    :func:`~blosc2.open`
    """
    # May we raise a DeprecationWarning here in the future?
    return pack_tensor(arr, chunksize=chunksize, urlpath=urlpath, **kwargs)


def load_array(urlpath: str, dparams: dict = None) -> np.ndarray:
    """Load a serialized NumPy array from a file.

    Parameters
    ----------
    urlpath: str
        The path to the file containing the serialized array.
    dparams: dict, optional
        A dictionary with the decompression parameters, which can
        be used in the :func:`~blosc2.decompress2` function.

    Returns
    -------
    out: np.ndarray
        The deserialized NumPy array.

    Raises
    ------
    TypeError
        If :paramref:`urlpath` is not in cframe format
    RunTimeError
        If any other error is detected.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(1e6)
    >>> serial_size = blosc2.save_array(a, "test.bl2", mode="w")
    >>> serial_size < a.size * a.itemsize
    True
    >>> a2 = blosc2.load_array("test.bl2")
    >>> np.array_equal(a, a2)
    True

    See also
    --------
    :func:`~blosc2.save_array`
    :func:`~blosc2.load_tensor`
    :func:`~blosc2.pack_array2`
    :func:`~blosc2.pack_tensor`
    """
    # May we raise a DeprecationWarning here in the future?
    return load_tensor(urlpath, dparams=dparams)


def pack_tensor(
    tensor: tensorflow.Tensor | torch.Tensor | np.ndarray, chunksize: int = None, **kwargs: dict
) -> bytes | int:
    """Pack (compress) a TensorFlow or PyTorch tensor or a NumPy array.

    Parameters
    ----------
    tensor: tensor or np.ndarray.
        The tensor or array to be packed.

    chunksize: int
        The size (in bytes) for the chunks during compression. If not provided,
        it is computed automatically.

    Other parameters
    ----------------
    kwargs: dict, optional
        These are the same as the kwargs in :func:`SChunk.__init__ <blosc2.schunk.SChunk.__init__>`.

    Returns
    -------
    out: bytes | int
        The serialized version (cframe) of the array.
        If urlpath is provided, the number of bytes in file is returned instead.

    Examples
    --------
    >>> import numpy as np
    >>> th = np.arange(1e6, dtype=np.float32)
    >>> cframe = blosc2.pack_tensor(th)
    >>> if not os.getenv("BTUNE_TRADEOFF"):
    ...     assert len(cframe) < th.size * th.itemsize
    ...

    See also
    --------
    :func:`~blosc2.unpack_tensor`
    :func:`~blosc2.save_tensor`
    """
    arr = np.asarray(tensor)

    schunk = blosc2.SChunk(chunksize=chunksize, data=arr, **kwargs)

    # Guess the kind of tensor / array
    repr_tensor = repr(tensor)
    if "tensor" in repr_tensor:
        kind = "torch"
    elif "Tensor" in repr_tensor:
        kind = "tensorflow"
    elif "array" in repr_tensor:
        kind = "numpy"
    else:
        raise TypeError(f"Unrecognized tensor/array: {tensor!r}")

    # dtype encoding requires some care
    dtype = arr.dtype.descr if arr.dtype.kind == "V" else arr.dtype.str

    schunk.vlmeta["__pack_tensor__"] = (kind, arr.shape, dtype)

    if schunk.urlpath is None:
        return schunk.to_cframe()
    else:
        return os.stat(schunk.urlpath).st_size


def _unpack_tensor(schunk):
    kind, shape, dtype = schunk.vlmeta["__pack_tensor__"]
    out = np.empty(shape, dtype=dtype)
    schunk.get_slice(out=out)

    if kind == "torch":
        import torch

        th = torch.from_numpy(out)
    elif kind == "tensorflow":
        import tensorflow as tf

        th = tf.constant(out)
    elif kind == "numpy":
        th = out
    else:
        raise TypeError(f"Unrecognized tensor kind: {kind}")
    return th


def unpack_tensor(cframe: bytes) -> tensorflow.Tensor | torch.Tensor | np.ndarray:
    """Unpack (decompress) a packed TensorFlow or PyTorch via a cframe.

    Parameters
    ----------
    cframe: bytes
        The packed tensor to be restored.

    Returns
    -------
    out: tensor or np.ndarray
        The unpacked TensorFlow or PyTorch tensor or NumPy array.

    Raises
    ------
    TypeError
        If :paramref:`cframe` is not of type bytes, or not a cframe.
    RunTimeError
        If some other problem is detected.

    Examples
    --------
    >>> import os
    >>> import numpy as np
    >>> th = np.arange(1e3, dtype=np.float32)
    >>> cframe = blosc2.pack_tensor(th)
    >>> if not os.getenv("BTUNE_TRADEOFF"):
    ...     assert len(cframe) < th.size * th.itemsize
    ...
    >>> th2 = blosc2.unpack_tensor(cframe)
    >>> a = np.asarray(th)
    >>> a2 = np.asarray(th2)
    >>> np.array_equal(a, a2)
    True

    See also
    --------
    :func:`~blosc2.pack_tensor`
    :func:`~blosc2.save_tensor`
    """
    schunk = blosc2.schunk_from_cframe(cframe, False)
    return _unpack_tensor(schunk)


def save_tensor(
    tensor: tensorflow.Tensor | torch.Tensor | np.ndarray,
    urlpath: str,
    chunksize: int = None,
    **kwargs: dict,
) -> int:
    """Save a serialized PyTorch or TensorFlow tensor or NumPy array in `urlpath`.

    Parameters
    ----------
    tensor: tensor, np.ndarray
        The tensor or array to be saved.

    urlpath: str
        The file path where the tensor or array will be saved.

    chunksize: int
        The size (in bytes) for the chunks during compression. If not provided,
        it is computed automatically.

    Other parameters
    ----------------
    kwargs: dict, optional
        These are the same as the kwargs in :func:`SChunk.__init__ <blosc2.schunk.SChunk.__init__>`.

    Returns
    -------
    out: int
        The number of bytes of the saved tensor or array.

    Examples
    --------
    >>> import numpy as np
    >>> th = np.arange(1e6, dtype=np.float32)
    >>> serial_size = blosc2.save_tensor(th, "test.bl2", mode="w")
    >>> if not os.getenv("BTUNE_TRADEOFF"):
    ...     assert serial_size < th.size * th.itemsize
    ...

    See also
    --------
    :func:`~blosc2.load_tensor`
    :func:`~blosc2.pack_tensor`
    :func:`~blosc2.open`
    """
    return pack_tensor(tensor, chunksize=chunksize, urlpath=urlpath, **kwargs)


def load_tensor(urlpath: str, dparams: dict = None) -> tensorflow.Tensor | torch.Tensor | np.ndarray:
    """Load a serialized PyTorch or TensorFlow  tensor or NumPy array in `urlpath`.

    Parameters
    ----------
    urlpath: str
        The file where the tensor or array is to be loaded.

    Other parameters
    ----------------
    dparams: dict, optional
        A dictionary with the decompression parameters, which are the same that can
        be used in the :func:`~blosc2.decompress2` function.

    Returns
    -------
    out: tensor or ndarray
        The unpacked PyTorch or TensorFlow tensor or NumPy array.

    Raises
    ------
    TypeError
        If :paramref:`urlpath` is not in cframe format
    RunTimeError
        If some other problem is detected.

    Examples
    --------
    >>> import numpy as np
    >>> th = np.arange(1e6, dtype=np.float32)
    >>> size = blosc2.save_tensor(th, "test.bl2", mode="w")
    >>> if not os.getenv("BTUNE_TRADEOFF"):
    ...     assert size < th.size * th.itemsize
    ...
    >>> th2 = blosc2.load_tensor("test.bl2")
    >>> np.array_equal(th, th2)
    True

    See also
    --------
    :func:`~blosc2.save_tensor`
    :func:`~blosc2.pack_tensor`
    """
    schunk = blosc2.open(urlpath, dparams=dparams)
    return _unpack_tensor(schunk)


def set_compressor(codec: blosc2.Codec) -> int:
    """Set the compressor to be used. If this function is not
    called, then :py:obj:`blosc2.Codec.BLOSCLZ <Codec>` will be used.

    Parameters
    ----------
    codec: :class:`Codec`
        The compressor to be used.

    Returns
    -------
    out: int
        The code for the compressor (>=0).

    Raises
    ------
    ValueError
        If the compressor is not recognized, or there is not support for it.

    Notes
    -----
    The `compname` parameter in python-blosc API has been replaced by :paramref:`codec` , using `compname`
    as parameter or a string as a :paramref:`codec` value will not work.

    See also
    --------
    :func:`~blosc2.get_compressor`
    :func:`~blosc2.compressor_list`
    """
    return blosc2_ext.set_compressor(codec)


def free_resources() -> None:
    """Free possible memory temporaries and thread resources.

    Returns
    -------
    out: None

    Notes
    -----
    Blosc maintain a pool of threads waiting for work as well as some
    temporary space.  You can use this function to release these
    resources when you are not going to use Blosc for a long while.

    Examples
    --------
    >>> blosc2.free_resources()
    """
    blosc2_ext.free_resources()


def set_nthreads(nthreads: int) -> int:
    """Set the number of threads to be used during Blosc operation.

    Parameters
    ----------
    nthreads: int
        The number of threads to be used during Blosc operation.

    Returns
    -------
    out: int
        The previous number of used threads.

    Raises
    ------
    ValueError
        If :paramref:`nthreads` is larger than the maximum number of threads blosc can use.
        If :paramref:`nthreads` is not a positive integer.

    Notes
    -----
    The maximum number of threads for Blosc is :math:`2^{31} - 1`. In some
    cases Blosc gets better results if you set the number of threads
    to a value slightly below than your number of cores
    (via :func:`~blosc2.detect_number_of_cores`).

    Examples
    --------
    Set the number of threads to 2 and then to 1:

    >>> oldn = blosc2.set_nthreads(2)
    >>> blosc2.set_nthreads(1)
    2

    See also
    --------
    :attr:`~blosc2.nthreads`
    """
    rc = blosc2_ext.set_nthreads(nthreads)
    blosc2.nthreads = nthreads
    return rc


def compressor_list(plugins: bool = False) -> list:
    """
    Returns a list of compressors (codecs) available in the C library.

    Parameters
    ----------
    plugins: bool
        Whether to include plugins or not.

    Returns
    -------
    out: list
        The list of codec names.

    See also
    --------
    :func:`~blosc2.get_compressor`
    :func:`~blosc2.set_compressor`

    """
    cap = blosc2.GLOBAL_REGISTERED_CODECS_STOP if plugins else blosc2.DEFINED_CODECS_STOP
    return list(key for key in blosc2.Codec if key.value <= cap)


def set_blocksize(blocksize: int = 0) -> None:
    """
    Force the use of a specific blocksize.

    Parameters
    ----------
    blocksize: int
        The blocksize to use. If 0, an automatic blocksize will be used (the default).

    Returns
    -------
    out: None

    Notes
    -----
    This is a low-level function and is recommended for expert users only.

    Examples
    --------
    >>> blosc2.set_blocksize(512)
    >>> blosc2.set_blocksize(0)
    """
    blosc2_ext.set_blocksize(blocksize)


def clib_info(codec: blosc2.Codec) -> tuple:
    """Return info for compression libraries in C library.

    Parameters
    ----------
    codec: :class:`Codec`
        The compressor.

    Returns
    -------
    out: tuple
        The associated library name and version.

    Notes
    -----
    The `cname` parameter in python-blosc API has been replaced by :paramref:`codec` , using `cname`
    as parameter or a string as a :paramref:`codec` value will not work.
    """
    return blosc2_ext.clib_info(codec)


def get_clib(bytesobj: str | bytes) -> str:
    """
    Return the name of the compression library for Blosc :paramref:`bytesobj` buffer.

    Parameters
    ----------
    bytesobj: str or bytes
        The compressed buffer.

    Returns
    -------
    out: str
        The name of the compression library.
    """
    return blosc2_ext.get_clib(bytesobj).decode("utf-8")


def get_compressor() -> str:
    """Get the current compressor that is used for compression.

    Returns
    -------
    out: str
        The name of the compressor.

    See also
    --------
    :func:`~blosc2.set_compressor`
    :func:`~blosc2.compressor_list`

    """
    return blosc2_ext.get_compressor().decode("utf-8")


def set_releasegil(gilstate: bool) -> bool:
    """
    Sets a boolean on whether to release the Python global inter-lock (GIL)
    during c-blosc compress and decompress operations or not.  This defaults
    to False.

    Parameters
    ----------
    gilstate: bool
        True to release the GIL

    Returns
    -------
    out: bool
        The previous value of the Python global inter-lock (GIL).

    Notes
    -----
    Designed to be used with larger chunk sizes and a ThreadPool.  There is a
    small performance penalty with releasing the GIL that will more harshly
    penalize small block sizes.

    Examples
    --------
    >>> oldReleaseState = blosc2.set_releasegil(True)
    """
    gilstate = bool(gilstate)
    return blosc2_ext.set_releasegil(gilstate)


def detect_number_of_cores() -> int:
    """Detect the number of cores in this system.

    Returns
    -------
    out: int
        The number of cores in this system.
    """
    if "count" in blosc2.cpu_info:
        return blosc2.cpu_info["count"]
    return 1  # Default


# Dictionaries for the maps between compressor names and libs
codecs = compressor_list(plugins=True)
# Map for compression libraries and versions
clib_versions = {}
for codec in compressor_list(plugins=False):
    clib_versions[codec.name] = clib_info(codec)[1].decode("utf-8")


def os_release_pretty_name():
    for p in ("/etc/os-release", "/usr/lib/os-release"):
        try:
            with open(p) as f:
                for line in f:
                    name, _, value = line.rstrip().partition("=")
                    if name == "PRETTY_NAME":
                        if len(value) >= 2 and value[0] in "\"'" and value[0] == value[-1]:
                            value = value[1:-1]
                        return value
        except OSError:
            pass
    return None


def print_versions():
    """Print all the versions of software that python-blosc2 relies on."""
    print("-=" * 38)
    print(f"python-blosc2 version: {blosc2.__version__}")
    print(f"Blosc version: {blosc2.blosclib_version}")
    print(f"Codecs available (including plugins): {', '.join([codec.name for codec in codecs])}")
    print("Main codec library versions:")
    for clib in sorted(clib_versions.keys()):
        print(f"  {clib}: {clib_versions[clib]}")
    print(f"Python version: {sys.version}")
    (sysname, _nodename, release, version, machine, processor) = platform.uname()
    print(f"Platform: {sysname}-{release}-{machine} ({version})")
    if sysname == "Linux":
        distro = os_release_pretty_name()
        if distro:
            print(f"Linux dist: {distro}")
    if not processor:
        processor = "not recognized"
    print(f"Processor: {processor}")
    print(f"Byte-ordering: {sys.byteorder}")
    # Internal Blosc threading
    print(f"Detected cores: {blosc2.ncores}")
    print(f"Number of threads to use by default: {blosc2.nthreads}")
    print("-=" * 38)


def apple_silicon_cache_size(cache_level: int) -> int:
    """Get the data cache_level size in bytes for Apple Silicon in MacOS.

    Apple Silicon has two clusters, Performance (0) and Efficiency (1).
    This function returns the data cache size for the Performance cluster.
    """
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    size = ctypes.c_size_t()
    if cache_level == 1:
        # We are interested in the L1 *data* cache size
        hwcachesize = "hw.perflevel0.l1dcachesize"
    else:
        hwcachesize = f"hw.perflevel0.l{cache_level}cachesize"
    hwcachesize = hwcachesize.encode("ascii")
    libc.sysctlbyname(hwcachesize, ctypes.byref(size), ctypes.byref(ctypes.c_size_t(8)), None, 0)
    return size.value


def linux_cache_size(cache_level: int, default_size: int) -> int:
    """Get the data cache_level size in bytes for Linux."""
    cache_size = default_size
    try:
        with open(f"/sys/devices/system/cpu/cpu0/cache/index{cache_level}/size") as f:
            size = f.read()
            if size.endswith("K\n"):
                cache_size = int(size[:-2]) * 1024
            elif size.endswith("M\n"):
                cache_size = int(size[:-2]) * 1024 * 1024
    except FileNotFoundError:
        # If the cache size cannot be read, return the default size
        pass
    return cache_size


def get_cpu_info():
    cpu_info = cpuinfo.get_cpu_info()
    # cpuinfo does not correctly retrieve the cache sizes for Apple Silicon, so do it manually
    if platform.system() == "Darwin":
        cpu_info["l1_data_cache_size"] = apple_silicon_cache_size(1)
        cpu_info["l2_cache_size"] = apple_silicon_cache_size(2)
        cpu_info["l3_cache_size"] = apple_silicon_cache_size(3)
    # cpuinfo does not correctly retrieve the cache sizes for all CPUs on Linux, so ask the kernel
    if platform.system() == "Linux":
        l1_data_cache_size = cpu_info.get("l1_data_cache_size", 32 * 1024)
        cpu_info["l1_data_cache_size"] = linux_cache_size(1, l1_data_cache_size)
        l2_cache_size = cpu_info.get("l2_cache_size", 256 * 1024)
        cpu_info["l2_cache_size"] = linux_cache_size(2, l2_cache_size)
        l3_cache_size = cpu_info.get("l3_cache_size", 1024 * 1024)
        cpu_info["l3_cache_size"] = linux_cache_size(3, l3_cache_size)
    return cpu_info


def get_blocksize() -> int:
    """Get the internal blocksize to be used during compression.

    Returns
    -------
    out: int
        The size in bytes of the internal block size.
    """
    return blosc2_ext.get_blocksize()


def get_cbuffer_sizes(src: object) -> tuple[(int, int, int)]:
    """
    Get the sizes of a compressed `src` buffer.

    Parameters
    ----------
    src: bytes-like object
        A compressed buffer. Must be a bytes-like object
        that supports the Python Buffer Protocol, like bytes,
        bytearray, memoryview, or numpy.ndarray.

    Returns
    -------
    (nbytes, cbytes, blocksize): tuple
        A tuple with the `nbytes`, `cbytes` and `blocksize` for the
        `src` compressed buffer.
    """
    return blosc2_ext.cbuffer_sizes(src)


# Compute a decent value for chunksize based on L3 and/or heuristics
def get_chunksize(blocksize, l3_minimum=2**20, l3_maximum=2**26):
    # Find a decent default when L3 cannot be detected by cpuinfo
    # Based mainly in heuristics
    chunksize = blocksize
    if blocksize * 32 < l3_maximum:
        chunksize = blocksize * 32
    # Refine with L2/L3 measurements (not always possible)
    cpu_info = blosc2.cpu_info
    if "l3_cache_size" in cpu_info:
        l3_cache_size = cpu_info["l3_cache_size"]
        # cpuinfo sometimes returns cache sizes as strings (like,
        # "4096 KB"), so refuse the temptation to guess and use the
        # value only when it is an actual int.
        # Also, sometimes cpuinfo does not return a correct L3 size;
        # so in general, enforcing L3 > L2 is a good sanity check.
        if isinstance(l3_cache_size, int) and l3_cache_size > 0:
            l2_cache_size = cpu_info.get("l2_cache_size", "Not found")
            if isinstance(l2_cache_size, int) and l3_cache_size > l2_cache_size:
                chunksize = l3_cache_size
    # Chunksize should be at least the size of L2
    l2_cache_size = cpu_info.get("l2_cache_size", "Not found")
    if isinstance(l2_cache_size, int) and l2_cache_size > chunksize:
        chunksize = l2_cache_size

    # When evaluating expressions, it is convenient to keep chunks for all operands in L3 cache,
    # so let's divide by 4 (3 operands + result is a typical situation for moderately complex
    # expressions)
    chunksize //= 4

    # Ensure a minimum size
    if chunksize < l3_minimum:
        chunksize = l3_minimum

    # In Blosc2, the chunksize cannot be larger than 2 GB - BLOSC2_MAX_BUFFERSIZE
    if chunksize > 2**31 - blosc2.MAX_OVERHEAD:
        chunksize = 2**31 - blosc2.MAX_OVERHEAD

    return chunksize


def nearest_divisor(a, b):
    if a > 100_000:
        # When `a` is largish, use a faster algorithm that only goes downwards
        for i in range(b, 0, -1):
            if a % i == 0:
                return i

    # When `a` is smallish, use a more general algorithm that can find forwards and backwards
    # Get all divisors of `a`; use a generator to avoid creating a list
    divisors = (i for i in range(1, a + 1) if a % i == 0)
    # Find the divisor nearest to b
    return min(divisors, key=lambda x: abs(x - b))


# Compute chunks and blocks partitions
def compute_partition(nitems, maxshape, minpart=None):
    if 0 in maxshape:
        raise ValueError("shapes with 0 dims are not supported")
    if nitems == 0:
        raise ValueError("zero-sized partitions are not supported")

    # Increase dims starting from the latest
    max_items = nitems
    if minpart is None:
        minpart = [1] * len(maxshape)
    partition = [1] * len(maxshape)
    for i, (size, minsize) in enumerate(zip(reversed(maxshape), reversed(minpart), strict=True)):
        if max_items <= 1:
            break
        rsize = max(size, minsize)
        if rsize <= max_items:
            rsize = rsize if size % rsize == 0 else nearest_divisor(size, rsize)
            partition[-(i + 1)] = rsize
        else:
            rsize = max(max_items, minsize)
            new_rsize = rsize if size % rsize == 0 else nearest_divisor(size, rsize)
            # If the new rsize is not too far from the original rsize, use it
            if rsize // 2 < new_rsize < rsize * 2:
                rsize = new_rsize
            partition[-(i + 1)] = rsize
        max_items //= rsize

    return partition


def compute_chunks_blocks(
    shape: tuple[int] | list,
    chunks: tuple | list | None = None,
    blocks: tuple | list | None = None,
    dtype: np.dtype = np.uint8,
    **kwargs: dict,
) -> tuple[(int, int)]:
    """
    Compute educated guesses for chunks and blocks of a :ref:`NDArray`.

    Parameters
    ----------
    shape: tuple or list
        The shape of the array.
    chunks: tuple or list
        The shape of the chunk.  If None, a guess is computed based on cache sizes
        and heuristics.
    blocks: tuple or list
        The shape of the block.  If None, a guess is computed based on cache sizes
        and heuristics.
    dtype: np.dtype
        The dtype of the array.
    kwargs: dict
        Other keyword arguments supported by the
        :obj:`SChunk.__init__ <blosc2.schunk.SChunk.__init__>` constructor.

    Returns
    -------
    tuple
        A (chunks, blocks) tuple with the computed guesses for chunks and blocks.
    """

    # Return an arbitrary value for chunks and blocks when shape has any 0 dim
    if 0 in shape:
        return (1,) * len(shape), (1,) * len(shape)

    if blocks:
        if not isinstance(blocks, tuple | list):
            blocks = [blocks]
        if len(blocks) != len(shape):
            raise ValueError("blocks should have the same length than shape")
        for block, dim in zip(blocks, shape, strict=True):
            if block == 0:
                raise ValueError("blocks cannot contain 0 dimension")
            if dim == 1 and block > dim:
                raise ValueError("blocks cannot be greater than shape if it is 1")
    if chunks:
        if not isinstance(chunks, tuple | list):
            chunks = [chunks]
        if len(chunks) != len(shape):
            raise ValueError("chunks should have the same length than shape")
        for chunk, dim in zip(chunks, shape, strict=True):
            if dim == 1 and chunk > dim:
                raise ValueError("chunks cannot be greater than shape if it is 1")

    if chunks is not None and blocks is not None:
        for block, chunk in zip(blocks, chunks, strict=True):
            if block > chunk:
                raise ValueError("blocks cannot be greater than chunks")
        return chunks, blocks

    cparams = kwargs.get("cparams") or copy.deepcopy(blosc2.cparams_dflts)
    # Typesize in dtype always has preference over typesize in cparams
    itemsize = cparams["typesize"] = np.dtype(dtype).itemsize

    if blocks is None:
        # Get the default blocksize for the compression params
        # Using an 8 MB buffer should be enough for detecting the whole range of blocksizes
        nitems = 2**23 // itemsize
        # compress2 is used just to provide a hint on the blocksize
        # However, it does not work well with filters that are not shuffle or bitshuffle,
        # so let's get rid of them
        filters = cparams.get("filters", None)
        if filters:
            cparams2 = copy.deepcopy(cparams)
            for i, filter in enumerate(filters):
                if filter not in (blosc2.Filter.SHUFFLE, blosc2.Filter.BITSHUFFLE):
                    cparams2["filters"][i] = blosc2.Filter.NOFILTER
        else:
            cparams2 = cparams
        # Force STUNE to get a hint on the blocksize
        aux_tuner = cparams2.get("tuner", blosc2.Tuner.STUNE)
        cparams2["tuner"] = blosc2.Tuner.STUNE
        src = blosc2.compress2(np.zeros(nitems, dtype=f"V{itemsize}"), **cparams2)
        _, _, blocksize = blosc2.get_cbuffer_sizes(src)
        # Maximum blocksize calculation
        max_blocksize = blocksize
        if platform.machine() == "x86_64":
            # For modern Intel/AMD archs, experiments say to use half of the L2 cache size
            max_blocksize = blosc2.cpu_info["l2_cache_size"] // 2
        elif platform.system() == "Darwin" and "arm" in platform.machine():
            # For Apple Silicon, experiments say to use half of the L1 cache size
            max_blocksize = blosc2.cpu_info["l1_data_cache_size"] // 2
        if "clevel" in cparams and cparams["clevel"] == 0:
            # Experiments show that, when no compression is used, it is not a good idea
            # to exceed half of private cache for the blocksize because speed suffers
            # too much during evaluations.
            blocksize = max_blocksize
        elif blocksize > max_blocksize:
            blocksize = max_blocksize

        cparams2["tuner"] = aux_tuner
    else:
        blocksize = math.prod(blocks) * itemsize

    # Now that a sensible blocksize has been computed, let's compute the blocks
    if chunks is None:
        maxshape = shape
    else:
        maxshape = [min(els) for els in zip(chunks, shape, strict=True)]
    blocks = compute_partition(blocksize // itemsize, maxshape)

    # Finally, the chunks
    if chunks is None:
        blocksize = math.prod(blocks) * itemsize
        chunksize = get_chunksize(blocksize)
        chunks = compute_partition(chunksize // itemsize, shape, blocks)

    return tuple(chunks), tuple(blocks)


def compress2(src: object, **kwargs: dict) -> str | bytes:
    """Compress :paramref:`src` with the given compression params (if given)

    Parameters
    ----------
    src: bytes-like object (supporting the buffer interface)
        The buffer to compress

    Other Parameters
    ----------------
    kwargs: dict, optional
        Compression parameters. The default values are in :class:`blosc2.CParams`.
        Keyword arguments supported:

            cparams: :class:`blosc2.CParams` or dict
                All the compression parameters that you want to use as
                a :class:`blosc2.CParams` or dict instance.
            others: Any
                If `cparams` is not passed, all the parameters of a :class:`blosc2.CParams`
                can be passed as keyword arguments.

    Returns
    -------
    out: str or bytes
        The compressed data in form of a Python str or bytes object.

    Raises
    ------
    RuntimeError
        If the data cannot be compressed into `dst`.
        If an internal error occurred, probably because some
        parameter is not a valid parameter.
    """
    if kwargs is not None and "cparams" in kwargs:
        if len(kwargs) > 1:
            raise AttributeError("Cannot pass both cparams and other kwargs already included in CParams")
        if isinstance(kwargs.get("cparams"), blosc2.CParams):
            kwargs = asdict(kwargs.get("cparams"))
        else:
            kwargs = kwargs.get("cparams")

    return blosc2_ext.compress2(src, **kwargs)


def decompress2(src: object, dst: object | bytearray = None, **kwargs: dict) -> str | bytes:
    """Compress :paramref:`src` with the given compression params (if given)

    Parameters
    ----------
    src: bytes-like object
        The data to be decompressed. Must be a bytes-like object
        that supports the Python Buffer Protocol, like bytes,
        bytearray, memoryview, or numpy.ndarray.
    dst: NumPy object or bytearray
        The destination NumPy object or bytearray to fill, the length
        of which must be greater than 0. The user must make sure
        that it has enough capacity for hosting the decompressed
        data. Default is `None`, meaning that a new bytes object
        is created, filled and returned.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Decompression parameters. The default values are in :class:`blosc2.DParams`.
        Keyword arguments supported:

            dparams: :class:`blosc2.DParams` or dict
                All the decompression parameters that you want to use as
                a :class:`blosc2.DParams` or dict instance.
            others: Any
                If `dparams` is not passed, all the parameters of a :class:`blosc2.DParams`
                can be passed as keyword arguments.

    Returns
    -------
    out: str or bytes
        The decompressed data in form of a Python str or bytes object if
        :paramref:`dst` is `None`. Otherwise, it will return `None` because the result
        will already be in :paramref:`dst`.

    Raises
    ------
    RuntimeError
        If the data cannot be compressed into :paramref:`dst`.
        If an internal error occurred, probably because some
        parameter is not a valid one.
        If :paramref:`dst` is `None` and could not create a bytes object to store the result.
    TypeError
        If :paramref:`src` does not support the Buffer Protocol.
    ValueError
        If the length of :paramref:`src` is smaller than the minimum.
        If :paramref:`dst` is not None and its length is 0.
    """
    if kwargs is not None and "dparams" in kwargs:
        if len(kwargs) > 1:
            raise AttributeError("Cannot pass both dparams and other kwargs already included in DParams")
        if isinstance(kwargs.get("dparams"), blosc2.DParams):
            kwargs = asdict(kwargs.get("dparams"))
        else:
            kwargs = kwargs.get("dparams")

    return blosc2_ext.decompress2(src, dst, **kwargs)


# Directory utilities
def remove_urlpath(path: str) -> None:
    """Permanently remove the file or the directory given by :paramref:`path`. This function is used during
    the tests of a persistent SChunk to remove it.

    Parameters
    ----------
    path: str
        The path of the directory or file.

    Returns
    -------
    out: None
    """
    if path is not None:
        if isinstance(path, pathlib.PurePath):
            path = str(path)
        path = path.encode("utf-8") if isinstance(path, str) else path
        blosc2_ext.remove_urlpath(path)


def schunk_from_cframe(cframe: bytes | str, copy: bool = False) -> blosc2.SChunk:
    """Create a :ref:`SChunk <SChunk>` instance out of a contiguous frame buffer.

    Parameters
    ----------
    cframe: bytes or str
        The bytes object containing the in-memory cframe.
    copy: bool
        Whether to internally do a copy or not. If `False`,
        the user is responsible for keeping a reference to `cframe`.

    Returns
    -------
    out: :ref:`SChunk <SChunk>`
        A new :ref:`SChunk <SChunk>` containing the data passed.

    See Also
    --------
    :func:`~blosc2.schunk.SChunk.to_cframe`

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> nchunks = 4
    >>> chunk_size = 200 * 1000 * 4
    >>> data = np.arange(nchunks * chunk_size // 4, dtype=np.int32)
    >>> cparams = blosc2.CParams(typesize=4)
    >>> schunk = blosc2.SChunk(data=data, cparams=cparams)
    >>> serialized_schunk = schunk.to_cframe()
    >>> print(f"Serialized SChunk length: {len(serialized_schunk)} bytes")
    Serialized SChunk length: 14129 bytes
    >>> deserialized_schunk = blosc2.schunk_from_cframe(serialized_schunk)
    >>> start = 1000
    >>> stop = 1005
    >>> sl_bytes = deserialized_schunk[start:stop]
    >>> sl = np.frombuffer(sl_bytes, dtype=np.int32)
    >>> print("Slice from deserialized SChunk:", sl)
    Slice from deserialized SChunk: [1000 1001 1002 1003 1004]
    >>> expected_slice = data[start:stop]
    >>> print("Expected slice:", expected_slice)
    Expected slice: [1000 1001 1002 1003 1004]
    """
    return blosc2_ext.schunk_from_cframe(cframe, copy)


def ndarray_from_cframe(cframe: bytes | str, copy: bool = False) -> blosc2.NDArray:
    """Create a :ref:`NDArray <NDArray>` instance out of a contiguous frame buffer.

    Parameters
    ----------
    cframe: bytes or str
        The bytes object containing the in-memory cframe.
    copy: bool
        Whether to internally do a copy or not. If `False`,
        the user is responsible for keeping a reference to `cframe`.

    Returns
    -------
    out: :ref:`NDArray <NDArray>`
        A new :ref:`NDArray <NDArray>` containing the data passed.

    See Also
    --------
    :func:`~blosc2.NDArray.to_cframe`
    """
    return blosc2_ext.ndarray_from_cframe(cframe, copy)


def register_codec(
    codec_name: str,
    id: int,
    encoder: Callable[[np.ndarray[np.uint8], np.ndarray[np.uint8], int, blosc2.SChunk], int] = None,
    decoder: Callable[[np.ndarray[np.uint8], np.ndarray[np.uint8], int, blosc2.SChunk], int] = None,
    version: int = 1,
) -> None:
    """Register a user defined codec.

    Parameters
    ----------
    codec_name: str
        Name of the codec.
    id: int
        Codec id, must be between 160 and 255 (both included).
    encoder: Python function or None
        This will receive an input to compress as a ndarray of dtype uint8, an output to fill
        the compressed buffer in as a ndarray of dtype uint8, the codec meta and the `SChunk` instance.
        It must return the size of the compressed buffer in bytes.
        If None then the codec name indicates a dynamic plugin which must be installed.
    decoder: Python function or None
        This will receive an input to decompress as a ndarray of dtype uint8, an output to fill
        the decompressed buffer in as a ndarray of dtype uint8, the codec meta and the `SChunk` instance.
        It must return the size of the decompressed buffer in bytes.
        If None then the codec name indicates a dynamic plugin which must be installed.
    version: int
        Codec version. Default is 1.

    Returns
    -------
    out: None

    Notes
    -----
    * Cannot use multi-threading when using an user defined codec.

    * User defined codecs can only be used inside a `SChunk` instance.

    * Both encoder and encoder functions must be given (for a Python codec), or none (for
      a dynamic plugin).

    See Also
    --------
    :func:`register_filter`

    Examples
    --------
    .. code-block:: python

        # Define encoder and decoder functions
        def encoder(input, output, meta, schunk):
            # Check whether the data is an arange
            step = int(input[1] - input[0])
            res = input[1:] - input[:-1]
            if np.min(res) == np.max(res):
                output[0:4] = input[0:4]  # start
                n = step.to_bytes(4, sys.byteorder)
                output[4:8] = [n[i] for i in range(4)]
                return 8
            else:
                # Not compressible, tell Blosc2 to do a memcpy
                return 0


        def decoder1(input, output, meta, schunk):
            # For decoding we only have to worry about the arange case
            # (other cases are handled by Blosc2)
            output[:] = [input[0] + i * input[1] for i in range(output.size)]

            return output.size


        # Register codec
        codec_name = 'codec1'
        id = 180
        blosc2.register_codec(codec_name, id, encoder, decoder)
    """
    if id in blosc2.ucodecs_registry:
        raise ValueError("Id already in use")
    blosc2_ext.register_codec(codec_name, id, encoder, decoder, version)


def register_filter(
    id: int,
    forward: Callable[[np.ndarray[np.uint8], np.ndarray[np.uint8], int, blosc2.SChunk], None] = None,
    backward: Callable[[np.ndarray[np.uint8], np.ndarray[np.uint8], int, blosc2.SChunk], None] = None,
    name: str = None,
) -> None:
    """Register an user defined filter.

    Parameters
    ----------
    id: int
        Filter id, must be between 160 and 255 (both included).
    forward: Python function
        This will receive an input to apply the filter as a ndarray of dtype uint8, an output to fill
        as a ndarray of dtype uint8, the filter meta and the corresponding `SChunk` instance.
        If None then the filter name indicates a dynamic plugin which must be installed.
    backward: Python function
        This will receive an input as a ndarray of dtype uint8, an output to fill
        as a ndarray of dtype uint8, the filter meta and the `SChunk` instance.
        If None then the filter name indicates a dynamic plugin which must be installed.
    name: str
        The filter name.
        If both `forward`and `backward` are None, this parameter must be passed to correctly
        load the dynamic filter.
    Returns
    -------
    out: None

    Notes
    -----
    * Cannot use multi-threading when using an user defined filter.

    * User defined filters can only be used inside a `SChunk` instance.

    See Also
    --------
    :func:`register_codec`

    Examples
    --------
    .. code-block:: python

        # Define forward and backward functions
        def forward(input, output, meta, schunk):
            nd_input = input.view(dtype)
            nd_output = output.view(dtype)

            nd_output[:] = nd_input + 1


        def backward(input, output, meta, schunk):
            nd_input = input.view(dtype)
            nd_output = output.view(dtype)

            nd_output[:] = nd_input - 1


        # Register filter
        id = 160
        blosc2.register_filter(id, forward, backward)
    """
    if id in blosc2.ufilters_registry:
        raise ValueError("Id already in use")
    blosc2_ext.register_filter(id, forward, backward, name)
