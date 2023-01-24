#!/usr/bin/env python
import ctypes
import sys
import pathlib
import numpy as np
import numpy.ctypeslib as ctl
from typing import Any

libname = pathlib.Path(pathlib.Path().absolute(), "dists")
# print("libname: ", libname)

# Load the shared library into c types.
if sys.platform.startswith("win"):
    c_distlib = ctypes.CDLL(libname / "Distances.dll")
else:
    c_distlib = ctypes.CDLL(libname / "Distances.so")


# before coming into getMBD/getGEO:
# image should be in range 0-256, type uint8. same goes for labels.
# if image has multiple channels, it should be called on each channel separately? otherwise we need to figure out a good way of aggregating it/doing vectorized distance.
# myarray = np.array(image, dtype=np.uint8, order='C') #not sure about the order actually
#


c_distlib.GEO.argtypes = [
    ctl.ndpointer(ctypes.c_uint8, flags="aligned, contiguous"),
    ctl.ndpointer(ctypes.c_uint8, flags="aligned, contiguous"),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctl.ndpointer(ctypes.c_double, flags="aligned, contiguous, writeable"),
    ctypes.c_int,
]
c_distlib.MBD.argtypes = [
    ctl.ndpointer(ctypes.c_uint8, flags="aligned, contiguous"),
    ctl.ndpointer(ctypes.c_uint8, flags="aligned, contiguous"),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctl.ndpointer(ctypes.c_double, flags="aligned, contiguous, writeable"),
    ctypes.c_int,
]
c_distlib.MBD.restype = None
c_distlib.GEO.restype = None


def getMBD(
    image: np.ndarray, label: np.ndarray, neighborhood: str = "full"
) -> np.ndarray:  # np.ndarray[Any, np.dtype[np.int32]]) -> np.ndarray[Any, np.dtype[np.double]]
    image = np.ascontiguousarray(image, dtype=np.uint8)
    label = np.ascontiguousarray(label, dtype=np.uint8)
    image.setflags(align=True)
    label.setflags(align=True)

    neighType = {"full": [8, 26], "semi": [4, 18], "direct": [4, 6]}
    Ntype = 0
    sizez = 1
    if image.ndim == 3:
        # 3dim image, TODO: add channel handling <- can't be done directly, need to apply on each separately.
        sizez = label.shape[2]
        Ntype = 1
    sizex, sizey = label.shape[0:2]
    classes = label.shape[-1] if (image.ndim + 1) == label.ndim else 1

    # fear of memory problems; is it safe to do dt=3D, then put dt[...,i] to MBD funct? will indexing break?
    # to be on the safe side, let's do each class separately, then join:

    dt = [np.zeros((sizex, sizey, sizez), dtype=np.double) for i in range(classes)]
    for i in range(classes):
        c_distlib.MBD(label, image, sizex, sizey, sizez, dt[i], neighType[neighborhood][Ntype])

    return np.stack(
        dt, axis=-1
    ).squeeze()  # here and above I assume that input label image has classes as last dim.


def getGEO(
    image: np.ndarray, label: np.ndarray, neighborhood: str = "full"
) -> np.ndarray:  # np.ndarray[Any, np.dtype[np.int32]]) -> np.ndarray[Any, np.dtype[np.double]]
    image = np.ascontiguousarray(image, dtype=np.uint8)
    label = np.ascontiguousarray(label, dtype=np.uint8)
    image.setflags(align=True)
    label.setflags(align=True)

    neighType = {"full": [8, 26], "semi": [4, 18], "direct": [4, 6]}
    Ntype = 0
    sizez = 1
    if image.ndim == 3:  # 3dim image, TODO: add channel handling
        sizez = label.shape[2]
        Ntype = 1
    sizex, sizey = label.shape[0:2]
    classes = label.shape[-1] if (image.ndim + 1) == label.ndim else 1

    # fear of memory problems; is it safe to do dt=3D, then put dt[...,i] to MBD funct? will indexing break?
    # to be on the safe side, let's do each class separately, then join:

    dt = [np.zeros((sizex, sizey, sizez), dtype=np.double) for i in range(classes)]
    for i in range(classes):
        c_distlib.GEO(label, image, sizex, sizey, sizez, dt[i], neighType[neighborhood][Ntype])

    return np.stack(
        dt, axis=-1
    ).squeeze()  # here and above I assume that input label image has classes as last dim.
