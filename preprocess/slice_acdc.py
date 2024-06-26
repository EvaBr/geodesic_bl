#!/usr/bin/env python3.10

import re
import pickle
import random
import argparse
import warnings
from pathlib import Path
from pprint import pprint
from functools import partial
from typing import Any, Callable, Tuple

import numpy as np
import nibabel as nib
from numpy import unique as uniq
from skimage.io import imsave
from skimage.transform import resize
# from PIL import Image

from utils import mmap_, uc_, map_, augment


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    return res.astype(np.uint8)


def get_frame(filename: str, regex: str = ".*_frame(\d+)(_gt)?\.nii.*") -> str:
    matched = re.match(regex, filename)

    if matched:
        return matched.group(1)
    raise ValueError(regex, filename)


def get_p_id(path: Path) -> str:
    '''
    The patient ID, for the ACDC dataset, is the folder containing the data.
    '''
    res = path.parent.name

    assert "patient" in res, res
    return res


def save_slices(img_p: Path, gt_p: Path,
                dest_dir: Path, shape: Tuple[int, int], n_augment: int,
                img_dir: str = "img", gt_dir: str = "gt") -> dict[str, tuple[float, float, float]]:
    p_id: str = get_p_id(img_p)
    assert "patient" in p_id
    assert p_id == get_p_id(gt_p)

    f_id: str = get_frame(img_p.name)
    assert f_id == get_frame(gt_p.name)

    # Load the data
    dx, dy, dz = nib.load(str(img_p)).header.get_zooms()
    assert dz in [5, 6.5, 7, 10], dz
    img = np.asarray(nib.load(str(img_p)).dataobj)
    gt = np.asarray(nib.load(str(gt_p)).dataobj)

    nx, ny = shape
    fx = nx / img.shape[0]
    fy = ny / img.shape[1]
    # print(f"Before dx {dx:.04f}, dy {dy:.04f}")
    dx /= fx
    dy /= fy
    # print(f"After dx {dx:.04f}, dy {dy:.04f}")

    assert img.shape == gt.shape
    # assert img.shape[:-1] == shape
    assert img.dtype in [np.uint8, np.int16, np.float32]

    # Normalize and check data content
    norm_img = norm_arr(img)  # We need to normalize the whole 3d img, not 2d slices
    assert 0 == norm_img.min() and norm_img.max() == 255, (norm_img.min(), norm_img.max())
    assert gt.dtype == norm_img.dtype == np.uint8

    resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)

    space_dict: dict[str, tuple[float, float, float]] = {}

    save_dir_img: Path = Path(dest_dir, img_dir)
    save_dir_gt: Path = Path(dest_dir, gt_dir)
    sizes_2d: np.ndarray = np.zeros(img.shape[-1])
    for j in range(img.shape[-1]):
        img_s = norm_img[:, :, j]
        gt_s = gt[:, :, j]
        assert img_s.shape == gt_s.shape
        assert gt_s.dtype == np.uint8

        # Resize and check the data are still what we expect
        r_img: np.ndarray = resize_(img_s, shape).astype(np.uint8)
        r_gt: np.ndarray = resize_(gt_s, shape, order=0)
        # r_gt: np.ndarray = np.array(Image.fromarray(gt_s, mode='L').resize(shape))
        assert set(uniq(r_gt)).issubset(set(uniq(gt))), (r_gt.dtype, uniq(r_gt))
        r_gt = r_gt.astype(np.uint8)
        assert r_img.dtype == r_gt.dtype == np.uint8
        assert 0 <= r_img.min() and r_img.max() <= 255  # The range might be smaller
        sizes_2d[j] = (r_gt == 3).astype(np.int64).sum()

        for k in range(n_augment + 1):
            if k == 0:
                a_img, a_gt = r_img, r_gt
            else:
                a_img, a_gt = map_(np.asarray, augment(r_img, r_gt))

            for save_dir, data in zip([save_dir_img, save_dir_gt], [a_img, a_gt]):
                filename = f"{p_id}_{f_id}_{k}_{j:02d}.png"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / filename

                space_dict[save_path.stem] = (dz, dy, dx)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    imsave(str(save_path), data)

    lv_gt = (gt == 3).astype(np.uint8)
    assert set(np.unique(lv_gt)) <= set([0, 1])
    assert lv_gt.shape == gt.shape

    lv_gt = resize_(lv_gt, (*shape, img.shape[-1]), order=0)
    assert set(np.unique(lv_gt)) <= set([0, 1])

    return space_dict


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the cleaning up is done before calling the script
    assert src_path.exists()
#    assert not dest_path.exists()

    # Get all the file names, avoid the temporal ones
    nii_paths: list[Path] = [p for p in src_path.rglob('*.nii.gz') if "_4d" not in str(p)]
    assert len(nii_paths) % 2 == 0, "Uneven number of .nii, one+ pair is broken"

    # We sort now, but also id matching is checked while iterating later on
    img_nii_paths: list[Path] = sorted(p for p in nii_paths if "_gt" not in str(p))
    gt_nii_paths: list[Path] = sorted(p for p in nii_paths if "_gt" in str(p))
    assert len(img_nii_paths) == len(gt_nii_paths) 
    assert (len(gt_nii_paths) == 200) or (len(gt_nii_paths) == 100) # Hardcode that value for sanity test
    paths: list[Tuple[Path, Path]] = list(zip(img_nii_paths, gt_nii_paths))

    print(f"Found {len(img_nii_paths)} pairs in total")
    pprint(paths[:5])

    pids: list[str] = sorted(set(map_(get_p_id, img_nii_paths)))
    # Sanity test: there is two scans per patients: we don't want to mix them up
    assert len(pids) == (len(img_nii_paths) // 2), (len(pids), len(img_nii_paths))

    random.shuffle(pids)  # Shuffle before to avoid any problem if the patients are sorted in any way
    fold_size: int = args.retains + args.retains_test
    offset: int = args.fold * fold_size
    # offset by (fold_size) at the beginning
    validation_slice = slice(offset, offset + args.retains)
    # offset by (fold_size + val_retains) at the beginning)
    test_slice = slice(offset + args.retains, offset + args.retains + args.retains_test)

    validation_pids: list[str] = pids[validation_slice]
    test_pids: list[str] = pids[test_slice]
    training_pids: list[str] = [pid for pid in pids if (pid not in validation_pids) and (pid not in test_pids)]

    assert len(validation_pids) == args.retains
    assert (len(validation_pids) + len(training_pids) + len(test_pids)) == len(pids)
    assert set(validation_pids).union(training_pids).union(test_pids) == set(pids)
    assert set(validation_pids).isdisjoint(training_pids)
    assert set(validation_pids).isdisjoint(test_pids)
    assert set(test_pids).isdisjoint(training_pids)

    # assert len(test_pids) == args.retains_test

    validation_paths: list[Tuple[Path, Path]] = [p for p in paths if get_p_id(p[0]) in validation_pids]
    test_paths: list[Tuple[Path, Path]] = [p for p in paths if get_p_id(p[0]) in test_pids]
    training_paths: list[Tuple[Path, Path]] = [p for p in paths if get_p_id(p[0]) in training_pids]

    # redundant sanity, but you never know
    assert set(validation_paths).isdisjoint(set(training_paths))
    assert set(validation_paths).isdisjoint(set(test_paths))
    assert set(test_paths).isdisjoint(set(training_paths))
    assert len(paths) == (len(validation_paths) + len(training_paths) + len(test_paths))
    assert len(validation_paths) == 2 * args.retains
    assert len(test_paths) == 2 * args.retains_test
    assert len(training_paths) == (len(paths) - 2 * fold_size)

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    for mode, _paths, n_augment in zip(["train", "val", "test"],
                                       [training_paths, validation_paths, test_paths],
                                       [args.n_augment, 0, 0]):
        if not _paths:
            continue
        img_paths, gt_paths = zip(*_paths)  # type: Tuple[Any, Any]

        dest_dir = Path(dest_path, mode)
        print(f"Slicing {len(img_paths)} pairs to {dest_dir}")
        assert len(img_paths) == len(gt_paths)

        pfun = partial(save_slices, dest_dir=dest_dir, shape=args.shape, n_augment=n_augment)
        all_spacedicts = mmap_(uc_(pfun), zip(img_paths, gt_paths))
        # for paths in tqdm(list(zip(img_paths, gt_paths)), ncols=50):
        #     uc_(pfun)(paths)

        for space_dict in all_spacedicts:
            resolution_dict |= space_dict

    if args.verbose:
        pprint(resolution_dict)

    with open(dest_path / "spacing_3d.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--img_dir', type=str, default="IMG")
    parser.add_argument('--gt_dir', type=str, default="GT")
    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--retains_test', type=int, default=0, help="Number of retained patient for the test data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--n_augment', type=int, default=0,
                        help="Number of augmentation to create per image, only for the training set")

    parser.add_argument('--verbose', action='store_true', help="Print more info (space dict at the end, for now).")
    args = parser.parse_args()
    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    args = get_args()
    random.seed(args.seed)

    main(args)
