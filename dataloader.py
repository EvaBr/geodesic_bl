#!/usr/env/bin python3.9

import io
import re
import pickle
import random
import torchvision.transforms.functional as TF
import collections.abc as container_abcs
from pathlib import Path
from itertools import repeat
from operator import itemgetter, mul
from functools import partial, reduce
from multiprocessing import cpu_count
from typing import Callable, BinaryIO, Match, Pattern, Tuple, Union, Optional, List, Dict

import torch
import numpy as np
from PIL import Image, ImageOps
from torch import Tensor
from torchvision import transforms
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, Sampler

from utils import map_, class2one_hot, one_hot2dist, id_
from utils import one_hot, depth



F = Union[Path, BinaryIO]
D = Union[Image.Image, np.ndarray, Tensor]


resizing_fn = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)


def png_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda img: img.convert('L'),
                lambda img: np.array(img)[np.newaxis, ...],
                lambda nd: nd / 255,  # max <= 1
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])
def png_transform_nonorm(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda img: img.convert('L'),
                lambda img: np.array(img)[np.newaxis, ...],
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

def liver_dt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda img: img[3:4, ...], #take only channel 3, as all other classes not needed
                lambda nd: torch.tensor(nd, dtype=torch.float32)[None, ...],
                lambda pol: torch.cat([-pol, pol], dim=1) #now we have bckg and liver channels
        ])
        
def liver_gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda img: np.array(img)[...],
                lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
                partial(class2one_hot, K=K),
                itemgetter(0),
                lambda cel: torch.cat([1.-cel[:, 3:4, ...], cel[:, 3:4, ...]], dim=1)
        ])

def equalized_png(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda img: img.convert('L'),
                lambda img: ImageOps.equalize(img),
                lambda img: np.array(img)[np.newaxis, ...],
                lambda nd: nd / 255,  # max <= 1
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])


def png_transform_npy(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda img: img.convert('L'),
                lambda img: np.array(img)[np.newaxis, ...],
                lambda nd: nd / 255,  # max <= 1
        ])


def npy_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda npy: np.array(npy)[np.newaxis, ...],
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])


def npy_crf_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda npy: np.array(npy),
                lambda nd: np.repeat(nd[None, :, :], 3, axis=0),
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])


def raw_npy_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda npy: np.array(npy),
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])


def from_numpy_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda nd: torch.tensor(nd)
        ])


def tensor_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])
def tensorT_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda nd: torch.tensor(nd, dtype=torch.float32),
                lambda dd: dd.unsqueeze(0),
                lambda dd: dd.transpose(0,-1),
                lambda nd: nd.squeeze()
        ])
def scaled_dt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda nd: 250*torch.tensor(nd)
        ])

def gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda img: np.array(img)[...],
                lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
                partial(class2one_hot, K=K),
                itemgetter(0)  # Then pop the element to go back to img shape
        ])


def dummy_gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                lambda img: np.array(img)[...],
                lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
                lambda t: torch.zeros_like(t),
                partial(class2one_hot, K=K),
                itemgetter(0)  # Then pop the element to go back to img shape
        ])


def dist_map_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
                gt_transform(resolution, K),
                lambda t: t.cpu().numpy(),
                partial(one_hot2dist, resolution=resolution),
                lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])


def unet_loss_weights_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        w_0: float = 10
        sigma: float = 5

        def closure(in_: D) -> Tensor:
                gt: Tensor = gt_transform(resolution, K)(in_)

                signed_dist_map: Tensor = dist_map_transform(resolution, K)(in_)
                dist_map: Tensor = torch.abs(signed_dist_map).type(torch.float32)

                w_c: Tensor = torch.einsum("k...->k", gt) / reduce(mul, gt.shape[1:])
                filled_w_c: Tensor = torch.einsum("k,k...->k...", w_c.type(torch.float32), torch.ones_like(dist_map))

                w: Tensor = filled_w_c + w_0 * torch.exp(- dist_map**2 / (2 * sigma**2))
                assert (K, *in_.shape) == w.shape == gt.shape, (in_.shape, w.shape, gt.shape)

                final: Tensor = torch.einsum("k...,k...->k...", gt.type(torch.float32), w)

                return final

        return closure


def get_loaders(args, data_folder: str,
                batch_size: int, n_class: int,
                debug: bool, in_memory: bool,
                dimensions: int,
                use_spacing: bool = False) -> Tuple[List[DataLoader], List[DataLoader]]:
        losses_list = eval(args.losses)
        if depth(losses_list) == 1:
                losses_list = [losses_list]

        list_folders_list = eval(args.folders)
        # For compatibility reasons, avoid changing all the previous configuration files
        if depth(list_folders_list) == 1:
                list_folders_list = [list_folders_list]
        # print(folders_list)

        augment = None
        if args.augment:
                augment = augmentation

        # Prepare the datasets and dataloaders
        print()
        train_loaders = []
        for i, (train_topfolder, folders_list) in \
                enumerate(zip(args.training_folders, list_folders_list)):

                folders, trans, are_hots = zip(*folders_list)
                print(f">> {i}th training loader: {train_topfolder} with {folders}")

                # Create partial functions: Easier for readability later (see the difference between train and validation)
                gen_dataset = partial(SliceDataset,
                                      transforms=trans,
                                      are_hots=are_hots,
                                      debug=debug,
                                      K=n_class,
                                      in_memory=in_memory,
                                      dimensions=dimensions,
                                      no_assert=args.no_assert_dataloader,
                                      ignore_norm=args.ignore_norm_dataloader)
                data_loader = partial(DataLoader,
                                      num_workers=min(cpu_count(), batch_size + 5),
                                      pin_memory=True,
                                      collate_fn=custom_collate)

                train_folders: List[Path] = [Path(data_folder, train_topfolder, f) for f in folders]
                # I assume all files have the same name inside their folder: makes things much easier
                train_names: List[str] = map_(lambda p: str(p.name), train_folders[0].glob("*"))
                t_spacing_p: Path = Path(data_folder, train_topfolder, "spacing.pkl")
                train_spacing_dict: Dict[str, Tuple[float, ...]] = pickle.load(open(t_spacing_p, 'rb')) if use_spacing else None
                train_set = gen_dataset(train_names,
                                        train_folders,
                                        augment=augment,
                                        spacing_dict=train_spacing_dict)
                if args.group_train:
                        train_sampler = PatientSampler(train_set, args.grp_regex, shuffle=True)
                        train_loader = data_loader(train_set,
                                                   batch_sampler=train_sampler)
                else:
                        train_loader = data_loader(train_set,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   drop_last=False)

                train_loaders.append(train_loader)

                if i == args.val_loader_id or (args.val_loader_id == -1 and (i + 1) == len(args.training_folders)):
                        print(f">> Validation dataloader (id {args.val_loader_id}), {train_topfolder} {folders}")
                        val_folders: List[Path] = [Path(data_folder, args.validation_folder, f) for f in folders]
                        val_names: List[str] = map_(lambda p: str(p.name), val_folders[0].glob("*"))
                        v_spacing_p: Path = Path(data_folder, args.validation_folder, "spacing.pkl")

                        val_spacing_dict: Dict[str, Tuple[float, ...]] 
                        val_spacing_dict = pickle.load(open(v_spacing_p, 'rb')) if use_spacing else None

                        val_set = gen_dataset(val_names,
                                              val_folders,
                                              spacing_dict=val_spacing_dict)
                        val_sampler = PatientSampler(val_set, args.grp_regex, shuffle=False) if args.group else None
                        val_batch_size = 1 if val_sampler else batch_size
                        val_loader = data_loader(val_set,
                                                 batch_sampler=val_sampler,
                                                 batch_size=val_batch_size)

        return train_loaders, [val_loader]


def augmentation(*args):  # hardcoded for aug in 2D!
        s = args[0].shape[-2:] #assume all tensors have same last two dims
        angle = 0
        if random.random() > 0.5:  # random rot
                angle = random.choice([-90, -45, 45, 90])

        i, j = 0, 0
        h, w = s[0], s[1]
        hmin, wmin = 150, 150  # how big the  random crops are at least
        hmax, wmax = 200, 200  # how big the  random crops are at most
        if random.random() > 0.25:  # random translate, crop, resize
                h, w = torch.randint(hmin, hmax+1, size=(1,)).item(), torch.randint(wmin, wmax+1, size=(1,)).item()
                i = torch.randint(0, s[0] - h, size=(1,)).item()
                j = torch.randint(0, s[1] - w, size=(1,)).item()

        #padding args:
        biggest_side = max(s)
        to_pad_x = int(biggest_side*1.5 - h)
        to_pad_y = int(biggest_side*1.5 - w)
        #padding for the left, top, right and bottom borders respectively
        padx0 = to_pad_x//2
        padx1 = to_pad_x - padx0
        pady0 = to_pad_y//2
        pady1 = to_pad_y - pady0
        
        #for img in args:
        #        cropped = TF.crop(img, i, j, h, w)
        #        padded = transforms.Pad((pady0, padx0, pady1, padx1), padding_mode="reflect")(cropped)
        #        rotated = TF.rotate(padded, angle)
        #        cropped2 = transforms.CenterCrop(s)(rotated)
        #        print(f"{img.shape=}\n {cropped.shape=}\n {padded.shape=}\n {rotated.shape=}\n {cropped2.shape=}\n")
                
        
        #F.pad(img, padding, fill, padding_mode)
        imges = [TF.center_crop(TF.rotate(
                        transforms.Pad((pady0, padx0, pady1, padx1), padding_mode="reflect")(TF.crop(img, i, j, h, w))
                        , angle), s) for img in args]
        #TODO: instead of padding, would it be better to resize? the question is then how to handle one-hot encoded gt -.-
        return imges


def get_test_loader(
        args,
        data_folder: str,
        batch_size: int,
        n_class: int,
        debug: bool,
        in_memory: bool,
        dimensions: int,
        use_spacing: bool = False,
        ) -> DataLoader:

        folders_list = eval(args.folders)  # just one level list. all assumed to reside in the same
        # data_folder/args.test_folder!

        # Prepare the datasets and dataloaders
        print()
        folders, trans, are_hots = zip(*folders_list)
        print(f">> Test dataloader, loading from {data_folder}/{args.test_folder}/: {folders}")

        gen_dataset = partial(
                SliceDataset,
                transforms=trans,
                are_hots=are_hots,
                debug=debug,
                K=n_class,
                in_memory=in_memory,
                dimensions=dimensions,
                no_assert=args.no_assert_dataloader,
                ignore_norm=args.ignore_norm_dataloader,
        )
        data_loader = partial(
                DataLoader,
                num_workers=min(cpu_count(), batch_size + 5),
                pin_memory=True,
                collate_fn=custom_collate,
        )

        test_folders: List[Path] = [Path(data_folder, args.test_folder, f) for f in folders]
        test_names: List[str] = map_(lambda p: str(p.name), test_folders[0].glob("*"))
        t_spacing_p: Path = Path(data_folder, args.test_folder, "spacing.pkl")

        test_spacing_dict: Dict[str, Tuple[float, ...]]
        test_spacing_dict = pickle.load(open(t_spacing_p, "rb")) if use_spacing else None

        test_set = gen_dataset(test_names, test_folders, spacing_dict=test_spacing_dict)
        test_sampler = PatientSampler(test_set, args.grp_regex, shuffle=False) if args.group else None
        test_batch_size = 1 if test_sampler else batch_size
        # <- should probably just assume grouping, since we need 3d metrics when testing

        test_loader = data_loader(test_set, batch_sampler=test_sampler, batch_size=test_batch_size)

        return test_loader


class SliceDataset(Dataset):
        def __init__(self, filenames: List[str], folders: List[Path], are_hots: List[bool],
                     transforms: List[Callable], debug=False, quiet=False,
                     K=4, in_memory: bool = False, spacing_dict: Dict[str, Tuple[float, ...]] = None,
                     augment: Optional[Callable] = None, ignore_norm: bool = False,
                     dimensions: int = 2, debug_size: int = 10, no_assert: bool = False) -> None:
                self.folders: List[Path] = folders
                self.transforms: List[Callable[[Tuple, int], Callable[[D], Tensor]]] = transforms
                assert len(self.transforms) == len(self.folders)

                self.are_hots: List[bool] = are_hots
                self.filenames: List[str] = filenames
                self.debug = debug
                self.K: int = K  # Number of classes
                self.in_memory: bool = in_memory
                self.quiet: bool = quiet
                self.spacing_dict: Optional[Dict[str, Tuple[float, ...]]] = spacing_dict
                if self.spacing_dict:
                        assert len(self.spacing_dict) == len(self.filenames)
                        print("> Spacing dictionary loaded correctly")

                self.augment: Optional[Callable] = augment
                self.ignore_norm: bool = ignore_norm
                self.dimensions: int = dimensions

                self.no_assert: bool = no_assert

                if self.debug:
                        self.filenames = self.filenames[:debug_size]

                assert self.check_files()  # Make sure all file exists

                if not self.quiet:
                        print(f">> Initializing {self.__class__.__name__} with {len(self.filenames)} images")
                        print(f"> {self.dimensions=}")
                        print([tr.__name__ for tr in self.transforms])
                        if self.augment:
                                print("> Will augment data online")

                # Load things in memory if needed
                self.files: List[List[F]] = SliceDataset.load_images(self.folders, self.filenames, self.in_memory)
                assert len(self.files) == len(self.folders)
                for files in self.files:
                        assert len(files) == len(self.filenames)

        def check_files(self) -> bool:
                for folder in self.folders:
                        if not Path(folder).exists():
                                print(f"{folder} DOES NOT EXIST!")
                                return False

                        for f_n in self.filenames:
                                if not Path(folder, f_n).exists():
                                        print(f"{folder}: {f_n} DOES NOT EXIST!")
                                        return False

                return True

        @staticmethod
        def load_images(folders: List[Path], filenames: List[str], in_memory: bool, quiet=False) -> List[List[F]]:
                def load(folder: Path, filename: str) -> F:
                        p: Path = Path(folder, filename)
                        if in_memory:
                                with open(p, 'rb') as data:
                                        res = io.BytesIO(data.read())
                                return res
                        return p
                if in_memory and not quiet:
                        print("> Loading the data in memory...")

                files: List[List[F]] = [[load(f, im) for im in filenames] for f in folders]

                return files

        def __len__(self):
                return len(self.filenames)

        def __getitem__(self, index: int) -> Dict[str, Union[str,
                                                             int,
                                                             Tensor,
                                                             List[Tensor],
                                                             List[Tuple[slice, ...]],
                                                             List[Tuple[Tensor, Tensor]]]]:
                filename: str = self.filenames[index]
                path_name: Path = Path(filename)
                images: List[D]

                if path_name.suffix == ".png":
                        images = [Image.open(files[index]) for files in self.files]
                elif path_name.suffix == ".npy":
                        images = [np.load(files[index]) for files in self.files]
                else:
                        raise ValueError(filename)

                resolution: Tuple[float, ...]
                if self.spacing_dict:
                        resolution = self.spacing_dict[path_name.stem]
                else:
                        resolution = tuple([1] * self.dimensions)

                # Final transforms and assertions
                assert len(images) == len(self.folders) == len(self.transforms)
                t_tensors: List[Tensor] = [tr(resolution, self.K)(e) for (tr, e) in zip(self.transforms, images)]
                _, *img_shape = t_tensors[0].shape
               # print([t.shape for t in t_tensors])

                final_tensors: List[Tensor]
                if self.augment:
                        final_tensors = self.augment(*t_tensors)
                else:
                        final_tensors = t_tensors
                del t_tensors

                if not self.no_assert:
                        # main image is between 0 and 1
                        if not self.ignore_norm:
                                assert 0 <= final_tensors[0].min() and final_tensors[0].max() <= 1, \
                                        (final_tensors[0].min(), final_tensors[0].max())

                     #   for ttensor in final_tensors[1:]:  # Things should be one-hot or at least have the shape
                     #           assert ttensor.shape == (self.K, *img_shape), (ttensor.shape, self.K, *img_shape)
                        for ttensor, is_hot in zip(final_tensors, self.are_hots):  # All masks (ground truths) are class encoded
                                if is_hot:
                                        assert one_hot(ttensor, axis=0), torch.einsum("k...->...", ttensor)

                img, gt = final_tensors[:2]

                return {'filenames': filename,
                        'images': final_tensors[0],
                        'gt': final_tensors[1],
                        #'gt_pts': final_tensors[2],
                        'labels': final_tensors[2:], #this will contain gt_pts, if given in B_DATA in makefile. DONT FORGET TO USE FLAG --cmpute_on_pts IN THAT CASE!
                        'spacings': torch.tensor(resolution),
                        'index': index}


_use_shared_memory = True


def custom_collate(batch):
        """Collate function to handle dict from dataset dict[str, Union[str, Tensor, list[Tensor], list[slice]]]"""

        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.sparse.FloatTensor):
                return batch
        elif isinstance(elem, torch.Tensor):
                out = None
                # if torch.utils.data.get_worker_info() is not None:
                if _use_shared_memory:
                        # If we're in a background process, concatenate directly into a
                        # shared memory tensor to avoid an extra copy
                        numel = sum(x.numel() for x in batch)
                        storage = elem.storage()._new_shared(numel)
                        out = elem.new(storage).resize_(len(batch), *list(elem.size()))
                return torch.stack(batch, 0, out=out)
        elif isinstance(elem, np.ndarray):
                return np.stack(batch)
        elif isinstance(elem, str) or isinstance(elem, slice) or isinstance(elem, int):
                return batch
        elif isinstance(elem, container_abcs.Mapping):
                return {key: custom_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, list):
                if len(elem) == 0:
                        return batch

                if isinstance(elem[0], tuple):  # Handling for spacings
                        return batch

                transposed = zip(*batch)
                return [custom_collate(samples) for samples in transposed]

        raise TypeError(elem_type)


class PatientSampler(Sampler):
        def __init__(self, dataset: SliceDataset, grp_regex, shuffle=False, quiet=False) -> None:
                filenames: List[str] = dataset.filenames
                # Might be needed in case of escape sequence fuckups
                # self.grp_regex = bytes(grp_regex, "utf-8").decode('unicode_escape')
                assert grp_regex is not None
                self.grp_regex = grp_regex

                # Configure the shuffling function
                self.shuffle: bool = shuffle
                self.shuffle_fn: Callable = (lambda x: random.sample(x, len(x))) if self.shuffle else id_

                # print(f"Grouping using {self.grp_regex} regex")
                # assert grp_regex == "(patient\d+_\d+)_\d+"
                # grouping_regex: Pattern = re.compile("grp_regex")
                grouping_regex: Pattern = re.compile(self.grp_regex)

                stems: List[str] = [Path(filename).stem for filename in filenames]  # avoid matching the extension
                matches: List[Match] = map_(grouping_regex.match, stems)
                patients: List[str] = [match.group(1) for match in matches]

                unique_patients: List[str] = list(set(patients))
                assert len(unique_patients) < len(filenames)
                if not quiet:
                        print(f"Found {len(unique_patients)} unique patients out of {len(filenames)} images ; regex: {self.grp_regex}")

                self.idx_map: Dict[str, List[int]] = dict(zip(unique_patients, repeat(None)))
                for i, patient in enumerate(patients):
                        if not self.idx_map[patient]:
                                self.idx_map[patient] = []

                        self.idx_map[patient] += [i]
                # print(self.idx_map)
                assert sum(len(self.idx_map[k]) for k in unique_patients) == len(filenames)

                for pid in self.idx_map.keys():
                        self.idx_map[pid] = sorted(self.idx_map[pid], key=lambda i: filenames[i])

                # print("Patient to slices mapping done")

        def __len__(self):
                return len(self.idx_map.keys())

        def __iter__(self):
                values = list(self.idx_map.values())
                shuffled = self.shuffle_fn(values)
                return iter(shuffled)
