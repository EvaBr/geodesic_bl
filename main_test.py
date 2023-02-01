#!/usr/bin/env python3.9

import argparse
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, cast, List, Dict

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from dataloader import get_test_loader
from utils import probs2one_hot, probs2class
from utils import dice_coef, save_images, tqdm_, dice_batch


def setup(args) -> Tuple[Any, Any]:
    print("\n>>> Setting up")
    cpu: bool = args.cpu or not torch.cuda.is_available()
    device = torch.device("cpu") if cpu else torch.device("cuda")

    if args.weights:
        if cpu:
            net = torch.load(args.weights, map_location="cpu")
        else:
            net = torch.load(args.weights)
        print(f">> Restored weights from {args.weights} successfully.")
    else:
        raise "Need existing network weights!"
    net.to(device)

    return net, device


def do_forward_pass(
    net: Any,
    device: Any,
    loaders: DataLoader,
    K: int,
    savedir: str = "",
    metric_axis: List[int] = [1],
    temperature: float = 1,
) -> Tuple[Tensor, Tensor]:

    net.eval()

    total_iteration: int = len(loaders)  # U
    total_images: int = len(loaders.dataset)  # D

    all_dices: Tensor = torch.zeros((total_images, K), dtype=torch.float32, device=device)
    three_d_dices: Tensor = torch.zeros((total_iteration, K), dtype=torch.float32, device=device)

    done_img: int = 0
    done_batch: int = 0
    desc = ">>> Testing... "
    tq_iter = tqdm_(total=total_iteration, desc=desc)

    for data in loaders:
        image: Tensor = data["images"].to(device)
        target: Tensor = data["gt"].to(device)
        filenames: List[str] = data["filenames"]
        assert not target.requires_grad

        B, C, *_ = image.shape
        #   print(image.shape)

        # Forward
        pred_logits: Tensor = net(image)
        #   print(pred_logits.shape)
        pred_probs: Tensor = F.softmax(temperature * pred_logits, dim=1)
        #   print(pred_probs.shape)
        predicted_mask: Tensor = probs2one_hot(
            pred_probs.detach()
        )  # Used only for dice computation
        assert not predicted_mask.requires_grad

        sm_slice = slice(done_img, done_img + B)  # Values only for current batch

        dices: Tensor = dice_coef(predicted_mask, target)
        assert dices.shape == (B, K), (dices.shape, B, K)
        all_dices[sm_slice, ...] = dices

        three_d_DSC: Tensor = dice_batch(predicted_mask, target)
        assert three_d_DSC.shape == (K,)
        three_d_dices[done_batch] = three_d_DSC  # type: ignore

        # Save images
        if savedir:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                predicted_class: Tensor = probs2class(pred_probs)
                save_images(predicted_class, filenames, savedir, "test", 0)

        # Logging
        big_slice = slice(0, done_img + B)  # Value for current and previous batches

        dsc_dict: dict = {f"DSC{n}": all_dices[big_slice, n].mean() for n in metric_axis} | {
            f"3d_DSC{n}": three_d_dices[:done_batch, n].mean() for n in metric_axis
        }
        nice_dict = {k: f"{v:.3f}" for (k, v) in dsc_dict.items()}

        done_img += B
        done_batch += 1
        tq_iter.set_postfix(nice_dict)
        tq_iter.update(1)

    tq_iter.close()
    del image, target, pred_logits, pred_probs

    print(f"{desc} " + ", ".join(f"{k}={v}" for (k, v) in nice_dict.items()))

    return (all_dices.detach().cpu(), three_d_dices.detach().cpu())


def run(args: argparse.Namespace) -> Dict[str, Tensor]:
    n_class: int = args.n_class
    savedir: str = args.workdir

    net, device = setup(args)

    data_loaders: DataLoader = get_test_loader(
        args,
        args.dataset,
        args.batch_size,
        n_class,
        args.debug,
        args.in_memory,
        args.dimensions,
        args.use_spacing,
    )

    n_test: int = len(data_loaders.dataset)  # Number of images in dataset
    l_test: int = len(data_loaders)
    # Number of iteration per epc: different if batch_size > 1

    metrics: Dict[str, Tensor] = {
        "test_dice": torch.zeros((n_test, n_class)).type(torch.float32),
        "test_3d_dsc": torch.zeros((l_test, n_class)).type(torch.float32),
    }

    print("\n>>> Starting the eval on test set")
    with torch.no_grad():
        test_dice, test_3d_dsc = do_forward_pass(
            net,
            device,
            data_loaders,
            n_class,
            savedir=savedir,
            metric_axis=args.metric_axis,
            temperature=args.temperature,
        )

        # Sort and save the metrics
        for k in metrics:
            assert metrics[k].shape == eval(k).shape, (metrics[k].shape, eval(k).shape, k)
            metrics[k] = eval(k)

        for k, e in metrics.items():
            np.save(Path(savedir, f"{k}.npy"), e.cpu().numpy())

        cols = {
            "test_dice": [f"Dice_{m}" for m in range(n_class)],
            "test_3d_dsc": [f"Dice_3D_{m}" for m in range(n_class)],
        }
        df = pd.concat(
            [pd.DataFrame(v.numpy(), columns=cols[k]) for k, v in metrics.items()],
            axis=1,
        )

        df.to_csv(Path(savedir, args.csv), float_format="%.4f", index_label="subject")

    for metric in metrics:
        print(f"\t{metric}: {metrics[metric].mean(dim=0)}")

    del net
    del data_loaders
    torch.cuda.empty_cache()

    return metrics


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument(
        "--folders", type=str, required=True, help="List of list of (subfolder, transform, is_hot)"
    )
    parser.add_argument("--network", type=str, required=True, help="The network to use")
    parser.add_argument("--n_class", type=int, required=True)
    parser.add_argument(
        "--metric_axis",
        type=int,
        nargs="*",
        required=True,
        help="Classes to display metrics. \
                Display only the average of everything if empty",
    )

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--shift_crop", action="store_true")
    parser.add_argument("--in_memory", action="store_true")
    parser.add_argument("--use_spacing", action="store_true")
    parser.add_argument("--no_assert_dataloader", action="store_true")
    parser.add_argument("--ignore_norm_dataloader", action="store_true")
    parser.add_argument(
        "--group",
        action="store_true",
        help="Group the patient slices together for validation. \
                Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.",
    )
    parser.add_argument("--grp_regex", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1, help="Temperature for the softmax")
    parser.add_argument("--modalities", type=int, default=1)
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--weights", type=str, required=True, help="Stored weights to restore")
    parser.add_argument("--test_folder", type=str, default="test")
    args = parser.parse_args()

    if args.metric_axis == []:
        args.metric_axis = list(range(args.n_class))
    print("\n", args)

    return args


if __name__ == "__main__":
    run(get_args())
