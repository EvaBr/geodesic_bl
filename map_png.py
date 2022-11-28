#!/usr/bin/env python3.9

import argparse
from pathlib import Path
from pprint import pprint
from itertools import starmap
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageOps

from utils import mmap_, starmmap_
from utils import np_class2one_hot, one_hot2dist, dm_rasterscan


def to_distmap(sources: tuple[Path, Path], dest: Path) -> None:
        import torch
        import FastGeodis
        lamb = {"intensity": 1,
                "geodesic": .5,
                "euclidean": 0}[args.distmap_mode]

        labels, img = sources
        K: int = args.K

        filename: str = dest.stem
        topfolder: str = dest.parents[0].name
        root: Path = dest.parents[1]

        lab_arr: np.ndarray = np.asarray(Image.open(labels).convert(mode="L"))
        img_arr: np.ndarray = np.asarray(Image.open(img).convert(mode='L')).astype(np.float32)
        assert lab_arr.shape == img_arr.shape
        assert lab_arr.dtype == np.uint8

        lab_oh: np.ndarray = np_class2one_hot(lab_arr[None, ...], K)[0]
        assert lab_oh.shape == (K, *img_arr.shape), lab_oh.shape

        res: np.ndarray = np.zeros(lab_oh.shape, dtype=np.float32)

        neg_oh: np.ndarray = np.logical_not(lab_oh)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        img_torch = torch.from_numpy(img_arr)[None, None, :, :].to(device)
        for k in range(K):
                if lab_oh[k].any():
                        lab_torch = torch.from_numpy(lab_oh[None, [k], :, :]).to(device)
                        neg_torch = torch.from_numpy(neg_oh[None, [k], :, :]).to(device)
                        assert img_torch.shape == lab_torch.shape, (img_torch.shape, lab_torch.shape)

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

        np.save(dest, res)

        for k in range(K):
                png_dest: Path = root / f"{topfolder}_{k}" / f"{filename}.png"
                plt.imsave(png_dest, res[k], cmap='viridis')


# def to_distmap_orig(sources: tuple[Path, Path], dest: Path) -> None:
#         labels, img = sources
#         K: int = args.K

#         filename: str = dest.stem
#         topfolder: str = dest.parents[0].name
#         root: Path = dest.parents[1]

#         lab_arr: np.ndarray = np.asarray(Image.open(labels).convert(mode="L"))
#         img_arr: np.ndarray = np.asarray(Image.open(img).convert(mode='L')).astype(np.float32)
#         assert lab_arr.shape == img_arr.shape
#         assert lab_arr.dtype == np.uint8

#         lab_oh: np.ndarray = np_class2one_hot(lab_arr[None, ...], K)[0]
#         assert lab_oh.shape == (K, *img_arr.shape), lab_oh.shape

#         res: np.ndarray = np.zeros(lab_oh.shape, dtype=np.float32)

#         neg_oh: np.ndarray = np.logical_not(lab_oh)
#         dists: np.ndarray = dm_rasterscan(img_arr, np.concatenate([lab_oh, neg_oh]),
#                                           scaling_factor=args.scaling_factor, alpha=args.alpha)

#         max_value: float = dists.max() if args.norm_dist else 1.

#         for k in range(K):
#                 post_dist, neg_dist = dists[[k, k + K]]
#                 res[k, ...] = post_dist / max_value - neg_dist / max_value

#                 if args.distmap_negative:
#                         posmask = lab_oh[k]
#                         if not posmask.any():
#                                 w, h = img_arr.shape
#                                 Xs, Ys = np.mgrid[:w, :h]
#                                 Xs -= w // 2
#                                 Ys -= h // 2

#                                 res[k, ...] = (Xs**2 + Ys**2)**.5 + 1
#                                 res[k, ...] /= res[k, ...].max()

#                 if args.norm_dist:
#                         assert -1 <= res[k].min() and res[k].max() <= 1

#                 # negmask = neg_oh[k]
#                 # if posmask.any() and k >= 1:
#                 #         print(f"{post_dist.min()=}, {post_dist.max()=}, {neg_dist.min()=}, {neg_dist.max()=}")

#                 #         min_ = min(post_dist.min(), -neg_dist.max())
#                 #         max_ = max(post_dist.max(), -neg_dist.min())
#                 #         print(f"{min_=}, {max_=}")
#                 #         figs = [(posmask, "posmask", [0, 1]),
#                 #                 (post_dist, "post_dist", [min_, max_]),
#                 #                 (negmask, "negmask", [0, 1]),
#                 #                 (-neg_dist, "neg_dist", [min_, max_]),
#                 #                 (res[k], "res", [min_, max_])]

#                 #         _, axes = plt.subplots(nrows=1, ncols=len(figs))

#                 #         for axe, (im, title, (vmin, vmax)) in zip(axes, figs):
#                 #                 axe.set_title(title)
#                 #                 axe.imshow(im, vmin=vmin, vmax=vmax)
#                 #         plt.show()

#         np.save(dest, res)

#         for k in range(K):
#                 png_dest: Path = root / f"{topfolder}_{k}" / f"{filename}.png"
#                 plt.imsave(png_dest, res[k], cmap='viridis')


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
                plt.imsave(png_dest, res[k], cmap='viridis')


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
                ImageOps.pad(img, (nw, nh),
                             method=Image.Resampling.NEAREST, centering=(.5, .5)).save(path)


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
                padded = ImageOps.pad(img,
                                      (pw, ph),
                                      method=Image.Resampling.NEAREST,
                                      centering=(.5, .5))
                padded.resize((nw, nh), resample=Image.Resampling.NEAREST).save(path)


DICT_FN: dict[str, Callable] = {
    "to_distmap": to_distmap,
    "to_euclid": to_euclid,
    "to_npy": to_npy,
    "to_one_hot_npy": to_one_hot_npy,
    "center8pad": center8pad,
    "resize": resize
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
                # starmmap_(DICT_FN[args.mode], tqdm(list(zip(sources_imgs, dest_paths_noext))))
                list(starmap(DICT_FN[args.mode], tqdm(list(zip(sources_imgs, dest_paths_noext)))))
        else:
                mmap_(DICT_FN[args.mode], tqdm(sources_imgs))


def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Params to transform (in place or not) png files')
        parser.add_argument('--src', type=Path, nargs="+", required=True)
        parser.add_argument('--dest', type=Path, default=None)

        parser.add_argument('--mode', type=str, choices=DICT_FN.keys())

        parser.add_argument('-K', type=int, default=2)
        parser.add_argument('--distmap_mode', type=str,
                            choices=["euclidean", "geodesic", "intensity"],
                            default="euclidean")
        parser.add_argument('--distmap_negative', action="store_true")
        parser.add_argument('--scaling_factor', type=float, default=1)
        parser.add_argument('--alpha', type=float, default=1)
        parser.add_argument('--size', type=int, nargs=2, default=None)
        # parser.add_argument('--lamb', type=float, default=1)

        parser.add_argument('--norm_dist', action="store_true",
                            help="Normalize the signed distance map, while paying attention not to shift the sign.")
        parser.add_argument('--verbose', action="store_true")

        args = parser.parse_args()

        print(args)

        return args


if __name__ == "__main__":
        args = get_args()  # This way available to the whole scope
        main(args)
