#!/usr/bin/env python3.9

import argparse
import warnings
from pathlib import Path
from pprint import pprint
from itertools import starmap
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imsave
from PIL import Image, ImageOps

from utils import mmap_, starmmap_
from utils import np_class2one_hot, one_hot2dist, dm_rasterscan
from scipy import ndimage as ndi


def to_dmap(sources: tuple[Path, Path], dest: Path) -> None:
    if args.distmap_mode == "geodesic":
        from dists.dists import getGEO as getDist
    elif args.distmap_mode == "intensity":
        from dists.dists import getMBD as getDist
    else:
        raise "Only intensity(=MBD)/geodesic options allowed with to_dmap!"

    labels, img = sources
    K: int = args.K

    filename: str = dest.stem
    topfolder: str = dest.parents[0].name
    root: Path = dest.parents[1]

    lab_arr: np.ndarray = np.asarray(Image.open(labels).convert(mode="L"))
    img_arr: np.ndarray = np.asarray(Image.open(img).convert(mode="L")).astype(np.float32)
    assert lab_arr.shape == img_arr.shape
    assert lab_arr.dtype == np.uint8

    lab_oh: np.ndarray = np_class2one_hot(lab_arr[None, ...], K)[0]
    assert lab_oh.shape == (K, *img_arr.shape), lab_oh.shape

    res: np.ndarray = np.zeros(lab_oh.shape, dtype=np.float32)

    neg_oh: np.ndarray = np.logical_not(lab_oh)

    for k in range(K):
        if lab_oh[k].any():
            pos_dist = getDist(img_arr.squeeze(), lab_oh[[k], :, :].squeeze())
            # TODO: keep median filtrng or not?
            #                ndi.median_filter(img_arr.squeeze(), size=3), lab_oh[[k], :, :].squeeze()
            neg_dist = getDist(img_arr.squeeze(), neg_oh[[k], :, :].squeeze())
            #                ndi.median_filter(img_arr.squeeze(), size=3), neg_oh[[k], :, :].squeeze()
            res[k, :, :] = pos_dist - neg_dist

    if args.norm_dist:
        max_value: float = np.abs(res).max()
        res /= max_value

        assert -1 <= res.min() and res.max() <= 1
    #  res *= 200  # TODO REMOVE

    np.save(dest, res)

    for k in range(K):
        png_dest: Path = root / f"{topfolder}_{k}" / f"{filename}.png"
        plt.imsave(png_dest, res[k], cmap="viridis")


def to_distmap(sources: tuple[Path, Path], dest: Path) -> None:
    import torch
    import FastGeodis

    lamb = {"intensity": 1, "geodesic": 0.5, "euclidean": 0}[args.distmap_mode]

    labels, img = sources
    K: int = args.K

    filename: str = dest.stem
    topfolder: str = dest.parents[0].name
    root: Path = dest.parents[1]

    lab_arr: np.ndarray = np.asarray(Image.open(labels).convert(mode="L"))
    img_arr: np.ndarray = np.asarray(Image.open(img).convert(mode="L")).astype(np.float32)
    assert lab_arr.shape == img_arr.shape
    assert lab_arr.dtype == np.uint8

    lab_oh: np.ndarray = np_class2one_hot(lab_arr[None, ...], K)[0]
    assert lab_oh.shape == (K, *img_arr.shape), lab_oh.shape

    res: np.ndarray = np.zeros(lab_oh.shape, dtype=np.float32)

    neg_oh: np.ndarray = np.logical_not(lab_oh)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu" if (not torch.cuda.is_available() or args.distmap_mode == "mbd") else "cuda"

    img_torch = torch.from_numpy(img_arr)[None, None, :, :].to(device)
    for k in range(K):
        if lab_oh[k].any():
            lab_torch = torch.from_numpy(lab_oh[None, [k], :, :]).to(device)
            neg_torch = torch.from_numpy(neg_oh[None, [k], :, :]).to(device)
            assert img_torch.shape == lab_torch.shape, (
                img_torch.shape,
                lab_torch.shape,
            )

            pos_dist = FastGeodis.generalised_geodesic2d(img_torch, lab_torch, 1e10, lamb, 2)
            neg_dist = FastGeodis.generalised_geodesic2d(img_torch, neg_torch, 1e10, lamb, 2)
            res[k, :, :] = (neg_dist - pos_dist)[0, 0].cpu().numpy()
            # res[k, :, :] = neg_dist[0, 0].cpu().numpy()

    # if args.distmap_mode == "euclidean":
    #         sanity: np.ndarray = one_hot2dist(lab_oh)
    #         print(f"{sanity.min()=} {sanity.max()=} {res.min()=} {res.max()=}")

    if args.norm_dist:
        max_value: float = np.abs(res).max()
        res /= max_value

        assert -1 <= res.min() and res.max() <= 1
    # res *= 200  # TODO REMOVE

    np.save(dest, res)

    for k in range(K):
        png_dest: Path = root / f"{topfolder}_{k}" / f"{filename}.png"
        plt.imsave(png_dest, res[k], cmap="viridis")


def to_superpixel(sources: tuple[Path, Path], dest: Path) -> None:
    from skimage.segmentation import slic

    labels, img = sources
    K: int = args.K

    filename: str = dest.stem
    topfolder: str = dest.parents[0].name
    root: Path = dest.parents[1]

    lab_arr: np.ndarray = np.asarray(Image.open(labels).convert(mode="L"))
    img_arr: np.ndarray = np.asarray(Image.open(img).convert(mode="L")).astype(np.float32)
    assert lab_arr.shape == img_arr.shape
    assert lab_arr.dtype == np.uint8

    lab_oh: np.ndarray = np_class2one_hot(lab_arr[None, ...], K)[0]
    assert lab_oh.shape == (K, *img_arr.shape), lab_oh.shape

    cluster: np.ndarray = slic(
        img_arr,
        n_segments=150,
        channel_axis=None,
        compactness=0.1,
        convert2lab=False,
        slic_zero=True,
    )
    assert cluster.shape == img_arr.shape

    cluster_dest: Path = root / f"{topfolder}_raw" / f"{filename}.png"
    imsave(cluster_dest, cluster.astype(np.uint8))

    res: np.ndarray = np.zeros_like(lab_arr)
    for v in np.unique(lab_arr):
        for c in np.unique(cluster[lab_arr == v]):
            res[cluster == c] = v

    png_dest: Path = root / f"{topfolder}" / f"{filename}.png"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        imsave(png_dest, res.astype(np.uint8))


def to_euclid(sources: tuple[Path, Path], dest: Path) -> None:
    labels, img = sources
    K: int = args.K

    filename: str = dest.stem
    topfolder: str = dest.parents[0].name
    root: Path = dest.parents[1]

    lab_arr: np.ndarray = np.asarray(Image.open(labels).convert(mode="L"))
    lab_oh: np.ndarray = np_class2one_hot(lab_arr[None, ...], K)[0]

    res: np.ndarray = one_hot2dist(lab_oh)

    np.save(dest, res)

    for k in range(K):
        png_dest: Path = root / f"{topfolder}_{k}" / f"{filename}.png"
        plt.imsave(png_dest, res[k], cmap="viridis")


def to_one_hot_npy(sources: tuple[Path], dest: Path) -> None:
    labels = sources[0]
    K: int = args.K

    lab_arr: np.ndarray = np.asarray(Image.open(labels).convert(mode="L"))
    W, H = lab_arr.shape
    assert lab_arr.dtype == np.uint8

    lab_oh: np.ndarray = np_class2one_hot(lab_arr[None, ...], K)[0].astype(np.int64)
    assert lab_oh.shape == (K, W, H)
    assert lab_oh.dtype == np.int64, lab_oh.dtype

    np.save(dest, lab_oh)


def to_npy(s: tuple[Path], d: Path) -> None:
    np.save(d, np.asarray(Image.open(s[0]).convert(mode="L")))


def center8pad(sources: tuple[Path, ...]) -> None:
    images: list[Image] = [Image.open(file).convert(mode="L") for file in sources]

    w, h = images[0].size
    nw: int = ((w // 16) + 1) * 16
    nh: int = ((h // 16) + 1) * 16
    assert nw > w
    assert nh > h

    for img, path in zip(images, sources):
        ImageOps.pad(img, (nw, nh), method=Image.Resampling.NEAREST, centering=(0.5, 0.5)).save(
            path
        )


def resize(sources: tuple[Path, ...]) -> None:
    images: list[Image] = [Image.open(file).convert(mode="L") for file in sources]

    w, h = images[0].size
    # print(w, h)
    nw, nh = args.size

    r: float = w / h
    nr: float = nw / nh

    pw: int
    ph: int
    if r < nr:  # too long, pad top and bottom
        pw = w
        ph = int(w / nr)
    else:  # Too short, pad left and right
        ph = h
        pw = int(h * nr)

    for img, path in zip(images, sources):
        padded = ImageOps.pad(img, (pw, ph), method=Image.Resampling.NEAREST, centering=(0.5, 0.5))
        padded.resize((nw, nh), resample=Image.Resampling.NEAREST).save(path)


DICT_FN: dict[str, Callable] = {
    "to_distmap": to_distmap,
    "to_dmap": to_dmap,
    "to_euclid": to_euclid,
    "to_superpixel": to_superpixel,
    "to_npy": to_npy,
    "to_one_hot_npy": to_one_hot_npy,
    "center8pad": center8pad,
    "resize": resize,
}


def main(args: argparse.Namespace):
    src_paths: list[Path] = [Path(s) for s in args.src]

    # Doesn't check the filenames properly match -- be careful
    sources_imgs: list[tuple[Path, ...]] = list(zip(*[sorted(p.glob("*.png")) for p in src_paths]))
    print(f"Found {len(sources_imgs)} images to modify")

    if args.verbose:
        pprint(sources_imgs[:10])

    stems: list[str] = [e[0].stem for e in sources_imgs]
    dest_paths_noext: Optional[list[Path]] = None
    if args.dest:
        dest_paths_noext = [Path(args.dest) / stem for stem in stems]

    if dest_paths_noext:
        starmmap_(DICT_FN[args.mode], tqdm(list(zip(sources_imgs, dest_paths_noext))))
        # list(starmap(DICT_FN[args.mode], tqdm(list(zip(sources_imgs, dest_paths_noext)))))
    else:
        mmap_(DICT_FN[args.mode], tqdm(sources_imgs))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Params to transform (in place or not) png files")
    parser.add_argument("--src", type=Path, nargs="+", required=True)
    parser.add_argument("--dest", type=Path, default=None)

    parser.add_argument("--mode", type=str, choices=DICT_FN.keys())

    parser.add_argument("-K", type=int, default=2)
    parser.add_argument(
        "--distmap_mode",
        type=str,
        choices=["euclidean", "geodesic", "intensity"],
        default="euclidean",
    )
    parser.add_argument("--distmap_negative", action="store_true")
    parser.add_argument("--scaling_factor", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--size", type=int, nargs=2, default=None)
    # parser.add_argument('--lamb', type=float, default=1)

    parser.add_argument(
        "--norm_dist",
        action="store_true",
        help="Normalize the signed distance map, while paying attention not to shift the sign.",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    args = get_args()  # This way available to the whole scope
    main(args)
