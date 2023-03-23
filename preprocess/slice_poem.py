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
from utils import np_class2one_hot

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



def get_p_id(path: Path) -> str:
    '''
    The patient ID, for the POEM dataset, is the number after subj.
    '''
    res = re.findall(r'500[0-9]+', path.name)

    assert len(res)>0, res
    assert "500" in res[0], res
    return res[0]


def save_slices(fat_p: Path, wat_p: Path, gt_p: Path,
                dest_dir: Path, shape: Tuple[int, int], n_augment: int,
                img_dir: str = "img", gt_dir: str = "gt") -> dict[str, tuple[float, float, float]]:
    p_id: str = get_p_id(fat_p)
    assert "500" in p_id
    assert p_id == get_p_id(gt_p) == get_p_id(wat_p)

    # Load the data
    
    fat = np.load(fat_p)
    wat = np.load(wat_p)
    gt = np.load(gt_p) #NOT ONE HOT
    space_dict = {}

    assert gt.ndim == 3, gt.shape
    assert fat.shape == gt.shape == wat.shape, (gt.shape, wat.shape)
    assert fat.dtype in [np.float64, np.float32], fat.dtype
    #assert gt.dtype in [np.uint8, np.int16], gt.dtype
    gt = gt.astype(np.uint8)

    # Normalize and check data content
    norm_fat = norm_arr(fat)  # We need to normalize the whole 3d img, not 2d slices
    norm_wat = norm_arr(wat)  # We need to normalize the whole 3d img, not 2d slices
    assert 0 == norm_fat.min() == norm_wat.min() and norm_fat.max() == 255 == norm_wat.max(), (norm_fat.min(), norm_fat.max())
    assert gt.dtype == norm_fat.dtype == norm_wat.dtype == np.uint8

    save_dir_img: Path = Path(dest_dir, img_dir)
    save_dir_gt: Path = Path(dest_dir, gt_dir)
    assert norm_fat.shape[0] == norm_fat.shape[2] == 256, norm_fat.shape 
    for j in range(50,201): #fat.shape[-2]):
        fat_s = norm_fat[:, j, :].squeeze()
        wat_s = norm_wat[:, j, :].squeeze()
        gt_s = gt[:, j, :].squeeze()
        assert fat_s.shape == wat_s.shape == gt_s.shape
        assert gt_s.dtype == np.uint8

       
        for k in range(n_augment + 1):
            if k == 0:
                a_fat, a_wat, a_gt = fat_s, wat_s, gt_s
            else:
                a_fat, a_wat, a_gt = map_(np.asarray, augment(fat_s, wat_s, gt_s))

            mainname = f"subj{p_id}_{k}_{j:03d}"
            for save_dir, data, abb in zip([save_dir_img, save_dir_img, save_dir_gt], 
                                            [a_fat, a_wat, a_gt], ['_fat','_wat','']):
                filename = f"{mainname}{abb}.png"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / filename

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    imsave(str(save_path), data)
                
                space_dict[filename[:-4]] = (2.070313, 2.070313, 8.0)
            
            #now directly save also NPYs, bcs why not.
            gt_onehot_s = np_class2one_hot(gt_s[None, ...], 7)[0]
            npy_img_p = Path(dest_dir, img_dir+'_npy')
            npy_gt_p = Path(dest_dir, gt_dir+'_npy') 
            npy_img_p.mkdir(parents=True, exist_ok=True)
            npy_gt_p.mkdir(parents=True, exist_ok=True)

            img_s = np.stack([fat_s, wat_s], axis=0)
            np.save(npy_gt_p / (mainname+'.npy'), gt_onehot_s)
            np.save(npy_img_p  / (mainname+'.npy'), img_s)

    return space_dict


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the cleaning up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    gtpath = Path(src_path, 'segs')
    watpath = Path(src_path, 'wats')
    fatpath = Path(src_path, 'fats')
    # Get all the file names, avoid the temporal ones
    wat_paths: list[Path] = sorted([p for p in watpath.glob('*.npy')])
    fat_paths: list[Path] = sorted([p for p in fatpath.glob('*.npy')])
    gt_paths: list[Path] = sorted([p for p in gtpath.glob('*.npy')])
    assert (len(wat_paths)+len(fat_paths)+len(gt_paths)) % 3 == 0, "Uneven number of .npy, one+ triplet is broken"

    # We sort now, but also id matching is checked while iterating later on
    assert len(fat_paths) == len(wat_paths) == len(gt_paths) == 50  # Hardcode that value for sanity test
    paths: list[Tuple[Path, Path, Path]] = list(zip(fat_paths, wat_paths, gt_paths))

    print(f"Found {len(fat_paths)} triplets in total")
    pprint(paths[:5])

    pids: list[str] = sorted(set(map_(get_p_id, fat_paths)))

    random.shuffle(pids)  # Shuffle before to avoid any problem if the patients are sorted in any way
    validation_slice = slice(0, args.retains)
    test_slice = slice(args.retains, args.retains + args.retains_test)

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

    validation_paths: list[Tuple[Path, Path, Path]] = [p for p in paths if get_p_id(p[0]) in validation_pids]
    test_paths: list[Tuple[Path, Path, Path]] = [p for p in paths if get_p_id(p[0]) in test_pids]
    training_paths: list[Tuple[Path, Path, Path]] = [p for p in paths if get_p_id(p[0]) in training_pids]

    # redundant sanity, but you never know
    assert set(validation_paths).isdisjoint(set(training_paths))
    assert set(validation_paths).isdisjoint(set(test_paths))
    assert set(test_paths).isdisjoint(set(training_paths))
    assert len(paths) == (len(validation_paths) + len(training_paths) + len(test_paths))
    assert len(validation_paths) == args.retains
    assert len(test_paths) == args.retains_test

    resolution_dict: dict[str, tuple[float, float, float]] = {}

    for mode, _paths, n_augment in zip(["train", "val", "test"],
                                       [training_paths, validation_paths, test_paths],
                                       [args.n_augment, 0, 0]):
        if not _paths:
            continue
        fat_paths, wat_paths, gt_paths = zip(*_paths)  # type: Tuple[Any, Any, Any]

        dest_dir = Path(dest_path, mode)
        print(f"Slicing {len(fat_paths)} triplets to {dest_dir}")
        assert len(fat_paths) == len(gt_paths) == len(wat_paths)

        pfun = partial(save_slices, dest_dir=dest_dir, shape=args.shape, n_augment=n_augment)
        all_spacedicts = mmap_(uc_(pfun), zip(fat_paths, wat_paths, gt_paths))
        
        for space_dict in all_spacedicts:
            resolution_dict |= space_dict



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--img_dir', type=str, default="img") #where to save to, i think
    parser.add_argument('--gt_dir', type=str, default="gt")
    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retains', type=int, default=5, help="Number of retained patient for the validation data")
    parser.add_argument('--retains_test', type=int, default=10, help="Number of retained patient for the test data")
    parser.add_argument('--seed', type=int, default=0)
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
