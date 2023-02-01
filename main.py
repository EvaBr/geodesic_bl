#!/usr/bin/env python3.9

import argparse
import warnings
from pathlib import Path
from functools import reduce
from operator import add, itemgetter
from shutil import copytree, rmtree
from typing import Any, Callable, Optional, Tuple, cast, List, Dict

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from dataloader import get_loaders
from utils import map_
from utils import depth
from utils import probs2one_hot, probs2class
from utils import dice_coef, save_images, tqdm_, dice_batch


def setup(
    args, n_class: int
) -> Tuple[Any, Any, Any, List[List[Callable]], List[List[float]], Callable]:
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
        net_class = getattr(__import__("networks"), args.network)
        net = net_class(args.modalities, n_class).to(device)
        net.init_weights()
    net.to(device)

    optimizer: Any  # disable an error for the optmizer (ADAM and SGD not same type)
    if args.use_sgd:
        optimizer = torch.optim.SGD(
            net.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4
        )
    else:
        optimizer = torch.optim.Adam(
            net.parameters(), lr=args.l_rate, betas=(0.9, 0.99), amsgrad=False
        )

    # print(args.losses)
    list_losses = eval(args.losses)
    if (
        depth(list_losses) == 1
    ):  # For compatibility reasons, avoid changing all the previous configuration files
        list_losses = [list_losses]

    loss_fns: List[List[Callable]] = []
    for i, losses in enumerate(list_losses):
        print(f">> {i}th list of losses: {losses}")
        tmp: List[Callable] = []
        for loss_name, loss_params, _ in losses:
            loss_class = getattr(__import__("losses"), loss_name)
            loss_params |= {"device": device}  # , 'n_classes': args.n_class}
            tmp.append(loss_class(**loss_params))
        loss_fns.append(tmp)

    loss_weights: List[List[float]] = [map_(itemgetter(2), losses) for losses in list_losses]

    scheduler = getattr(__import__("scheduler"), args.scheduler)(**eval(args.scheduler_params))

    return net, optimizer, device, loss_fns, loss_weights, scheduler


def do_epoch(
    mode: str,
    net: Any,
    device: Any,
    loaders: List[DataLoader],
    epc: int,
    list_loss_fns: List[List[Callable]],
    list_loss_weights: List[List[float]],
    K: int,
    savedir: str = "",
    optimizer: Any = None,
    metric_axis: List[int] = [1],
    compute_3d_dice: bool = False,
    temperature: float = 1,
    compute_on_pts: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    assert mode in ["train", "val", "dual"]

    if mode == "train":
        net.train()
        desc = f">> Training   ({epc})"
    elif mode == "val":
        net.eval()
        desc = f">> Validation ({epc})"

    total_iteration: int = sum(len(loader) for loader in loaders)  # U
    total_images: int = sum(len(loader.dataset) for loader in loaders)  # D
    n_loss: int = max(map(len, list_loss_fns))

    all_dices: Tensor = torch.zeros((total_images, K), dtype=torch.float32, device=device)
    loss_log: Tensor = torch.zeros((total_iteration, n_loss), dtype=torch.float32, device=device)

    three_d_dices: Optional[Tensor]
    if compute_3d_dice and mode == "val":
        three_d_dices = torch.zeros((total_iteration, K), dtype=torch.float32, device=device)
    else:
        three_d_dices = None

    all_dices_pts: Optional[Tensor]
    if compute_on_pts:
        all_dices_pts = torch.zeros((total_images, K), dtype=torch.float32, device=device)
    else:
        all_dices_pts = None

    done_img: int = 0
    done_batch: int = 0
    tq_iter = tqdm_(total=total_iteration, desc=desc)
    for i, (loader, loss_fns, loss_weights) in enumerate(
        zip(loaders, list_loss_fns, list_loss_weights)
    ):
        for data in loader:
            # t0 = time()
            image: Tensor = data["images"].to(device)
            target: Tensor = data["gt"].to(device)
            filenames: List[str] = data["filenames"]
            assert not target.requires_grad
            labels: List[Tensor] = [e.to(device) for e in data["labels"]]
            if compute_on_pts:
                target_pts: Tensor = labels[0]
                assert not target_pts.requires_grad
                labels = labels[1:]
            B, C, *_ = image.shape
            #   print(image.shape)
            #   print([hik.shape for hik in labels])

            # Reset gradients
            if optimizer:
                optimizer.zero_grad()

            # Forward
            pred_logits: Tensor = net(image)
            #   print(pred_logits.shape)
            pred_probs: Tensor = F.softmax(temperature * pred_logits, dim=1)
            #   print(pred_probs.shape)
            predicted_mask: Tensor = probs2one_hot(
                pred_probs.detach()
            )  # Used only for dice computation
            assert not predicted_mask.requires_grad

            assert len(loss_fns) == len(loss_weights) == len(labels)
            ziped = zip(loss_fns, labels, loss_weights)
            losses = [w * loss_fn(pred_probs, label) for loss_fn, label, w in ziped]
            loss = reduce(add, losses)
            assert loss.shape == (), loss.shape

            # Backward
            if optimizer:
                loss.backward()
                optimizer.step()

            # Compute and log metrics
            for j in range(len(loss_fns)):
                loss_log[done_batch, j] = losses[j].detach()

            sm_slice = slice(done_img, done_img + B)  # Values only for current batch

            dices: Tensor = dice_coef(predicted_mask, target)
            assert dices.shape == (B, K), (dices.shape, B, K)
            all_dices[sm_slice, ...] = dices

            if compute_on_pts:
                dices_pts: Tensor = dice_coef(predicted_mask, target_pts)
                assert dices_pts.shape == (B, K), (dices_pts.shape, B, K)

                all_dices_pts[sm_slice, ...] = dices_pts

            if compute_3d_dice and mode == "val":
                three_d_DSC: Tensor = dice_batch(predicted_mask, target)
                assert three_d_DSC.shape == (K,)

                three_d_dices[done_batch] = three_d_DSC  # type: ignore

            # Save images
            if savedir:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    predicted_class: Tensor = probs2class(pred_probs)
                    save_images(predicted_class, filenames, savedir, mode, epc)

            # Logging
            big_slice = slice(0, done_img + B)  # Value for current and previous batches

            dsc_dict: dict = {f"DSC{n}": all_dices[big_slice, n].mean() for n in metric_axis} | (
                {f"3d_DSC{n}": three_d_dices[:done_batch, n].mean() for n in metric_axis}
                if three_d_dices is not None
                else {}
            )

            loss_dict = {f"loss_{i}": loss_log[:done_batch].mean(dim=0)[i] for i in range(n_loss)}

            stat_dict = dsc_dict | loss_dict
            nice_dict = {k: f"{v:.3f}" for (k, v) in stat_dict.items()}

            done_img += B
            done_batch += 1
            tq_iter.set_postfix({**nice_dict, "loader": str(i)})
            tq_iter.update(1)
    tq_iter.close()
    del image, target, labels, pred_logits, pred_probs

    print(f"{desc} " + ", ".join(f"{k}={v}" for (k, v) in nice_dict.items()))

    return (
        loss_log.detach().cpu(),
        all_dices.detach().cpu(),
        all_dices_pts.detach().cpu() if all_dices_pts is not None else None,
        three_d_dices.detach().cpu() if three_d_dices is not None else None,
    )


def run(args: argparse.Namespace) -> Dict[str, Tensor]:
    n_class: int = args.n_class
    lr: float = args.l_rate
    savedir: str = args.workdir
    n_epoch: int = args.n_epoch
    val_f: int = args.val_loader_id

    loss_fns: List[List[Callable]]
    BLids: List[int]
    loss_weights: List[List[float]]
    net, optimizer, device, loss_fns, loss_weights, scheduler = setup(args, n_class)
    BLids = [i for i, f in enumerate(loss_fns[val_f]) if f.__class__.__name__ == "SurfaceLoss"]
    train_loaders: List[DataLoader]
    val_loaders: List[DataLoader]
    train_loaders, val_loaders = get_loaders(
        args,
        args.dataset,
        args.batch_size,
        n_class,
        args.debug,
        args.in_memory,
        args.dimensions,
        args.use_spacing,
    )

    n_tra: int = sum(len(tr_lo.dataset) for tr_lo in train_loaders)  # Number of images in dataset
    l_tra: int = sum(
        len(tr_lo) for tr_lo in train_loaders
    )  # Number of iteration per epc: different if batch_size > 1
    n_val: int = sum(len(vl_lo.dataset) for vl_lo in val_loaders)
    l_val: int = sum(len(vl_lo) for vl_lo in val_loaders)
    n_loss: int = max(map(len, loss_fns))
    n_val_loss: int = len(loss_fns[val_f])

    best_dice: Tensor = cast(Tensor, torch.zeros(1).type(torch.float32))
    best_3d_dsc: Tensor = cast(Tensor, torch.zeros(1).type(torch.float32))
    best_epoch: int = 0
    metrics: Dict[str, Tensor] = {
        "val_dice": torch.zeros((n_epoch, n_val, n_class)).type(torch.float32),
        "val_loss": torch.zeros((n_epoch, l_val, n_val_loss)).type(torch.float32),
        "tra_dice": torch.zeros((n_epoch, n_tra, n_class)).type(torch.float32),
        "tra_loss": torch.zeros((n_epoch, l_tra, n_loss)).type(torch.float32),
    }
    if args.compute_3d_dice:
        metrics["val_3d_dsc"] = cast(
            Tensor, torch.zeros((n_epoch, l_val, n_class)).type(torch.float32)
        )

    if args.compute_on_pts:
        metrics["tra_dice_pts"] = cast(
            Tensor, torch.zeros((n_epoch, n_tra, n_class)).type(torch.float32)
        )
        metrics["val_dice_pts"] = cast(
            Tensor, torch.zeros((n_epoch, n_val, n_class)).type(torch.float32)
        )

    print("\n>>> Starting the training")
    for i in range(n_epoch):
        # Do training and validation loops
        tra_loss, tra_dice, tra_dice_pts, _ = do_epoch(
            "train",
            net,
            device,
            train_loaders,
            i,
            loss_fns,
            loss_weights,
            n_class,
            savedir=savedir if args.save_train else "",
            optimizer=optimizer,
            metric_axis=args.metric_axis,
            temperature=args.temperature,
            compute_on_pts=args.compute_on_pts,
        )
        with torch.no_grad():
            val_res = do_epoch(
                "val",
                net,
                device,
                val_loaders,
                i,
                [loss_fns[val_f]],
                [loss_weights[val_f]],
                n_class,
                savedir=savedir,
                metric_axis=args.metric_axis,
                compute_3d_dice=args.compute_3d_dice,
                temperature=args.temperature,
                compute_on_pts=args.compute_on_pts,
            )
            val_loss, val_dice, val_dice_pts, val_3d_dsc = val_res

        # Sort and save the metrics
        for k in metrics:
            assert metrics[k][i].shape == eval(k).shape, (metrics[k][i].shape, eval(k).shape, k)
            metrics[k][i] = eval(k)

        for k, e in metrics.items():
            np.save(Path(savedir, f"{k}.npy"), e.cpu().numpy())

        # df = pd.DataFrame({"tra_loss": metrics["tra_loss"].mean(dim=(1, 2)).numpy(),
        #                   "val_loss": metrics["val_loss"].mean(dim=(1, 2)).numpy(),
        #                   "tra_dice": metrics["tra_dice"][:, :, -1].mean(dim=1).numpy(),
        #                   "val_dice": metrics["val_dice"][:, :, -1].mean(dim=1).numpy()}) #only last class. not ok for multiclass. ?
        cols = {
            "tra_loss": [f"tra_Loss{m}" for m in range(n_loss)],
            "val_loss": [f"val_Loss{m}" for m in range(n_val_loss)],
            "tra_dice": [f"tra_Dice{m}" for m in range(n_class)],
            "val_dice": [f"val_Dice{m}" for m in range(n_class)],
            "tra_dice_pts": [f"tra_Dice_pts{m}" for m in range(n_class)],
            "val_dice_pts": [f"val_Dice_pts{m}" for m in range(n_class)],
            "val_3d_dsc": [f"val_3d_Dice{m}" for m in range(n_class)],
        }
        df = pd.concat(
            [
                pd.DataFrame(v.mean(dim=1).numpy().tolist(), columns=cols[k])
                for k, v in metrics.items()
            ],
            axis=1,
        )

        df.to_csv(Path(savedir, args.csv), float_format="%.4f", index_label="epoch")

        # Save model if better
        current_dice: Tensor = val_dice[:, args.metric_axis].mean()
        current_3d_dsc: Tensor = metrics["val_3d_dsc"][i, :, args.metric_axis].mean()
        if current_dice > best_dice:
            best_epoch = i
            best_dice = current_dice
            best_3d_dsc = current_3d_dsc

            with open(Path(savedir, "best_epoch.txt"), "w") as f:
                f.write(str(i))
            best_folder = Path(savedir, "best_epoch")
            if best_folder.exists():
                rmtree(best_folder)
            copytree(Path(savedir, f"iter{i:03d}"), Path(best_folder))
            torch.save(net, Path(savedir, "best.pkl"))

        # update scheduler if needed:
        current_val_bl_loss = val_loss[..., BLids].mean()
        if (
            scheduler.__class__.__name__ == "BLbasedWeight"
            and current_val_bl_loss < args.bl_thr
            and current_dice > 0
        ):
            # TODO instead: wait for BL to be below a threshold for 3 or so epochs?
            scheduler.stop = True
        optimizer, loss_fns, loss_weights = scheduler(i, optimizer, loss_fns, loss_weights)

        # if args.schedule and (i > (best_epoch + 20)):
        if args.schedule and (i % (best_epoch + 20) == 0):  # Yeah, ugly but will clean that later
            for param_group in optimizer.param_groups:
                lr *= 0.5
                param_group["lr"] = lr
                print(f">> New learning Rate: {lr}")

        if i > 0 and not (i % 5):
            maybe_3d = f", 3d_DSC: {best_3d_dsc:.3f}" if args.compute_3d_dice else ""  # TODO
            print(f">> Best results at epoch {best_epoch}: DSC: {best_dice:.3f}{maybe_3d}")

    # Because displaying the results at the end is actually convenient
    maybe_3d = f", 3d_DSC: {best_3d_dsc:.3f}" if args.compute_3d_dice else ""  # TODO
    print(f">> Best results at epoch {best_epoch}: DSC: {best_dice:.3f}{maybe_3d}")
    for metric in metrics:
        # Do not care about training values, nor the loss (keep it simple)
        if "val" in metric and "loss" not in metric:
            print(f"\t{metric}: {metrics[metric][best_epoch].mean(dim=0)}")

    del net
    del train_loaders
    del val_loaders
    torch.cuda.empty_cache()

    return metrics


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument(
        "--losses",
        type=str,
        required=True,
        help="List of list of (loss_name, loss_params, bounds_name, bounds_params, fn, weight)",
    )
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
    parser.add_argument("--in_memory", action="store_true")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--use_sgd", action="store_true")
    parser.add_argument("--compute_3d_dice", action="store_true")
    parser.add_argument("--compute_on_pts", action="store_true")
    parser.add_argument("--save_train", action="store_true")
    parser.add_argument("--use_spacing", action="store_true")
    parser.add_argument("--no_assert_dataloader", action="store_true")
    parser.add_argument("--ignore_norm_dataloader", action="store_true")
    parser.add_argument(
        "--group",
        action="store_true",
        help="Group the patient slices together for validation. \
                Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.",
    )
    parser.add_argument(
        "--group_train",
        action="store_true",
        help="Group the patient slices together for training. \
                Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.",
    )

    parser.add_argument("--n_epoch", nargs="?", type=int, default=200, help="# of the epochs")
    parser.add_argument("--l_rate", nargs="?", type=float, default=5e-4, help="Learning Rate")
    parser.add_argument("--grp_regex", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1, help="Temperature for the softmax")
    parser.add_argument("--scheduler", type=str, default="DummyScheduler")
    parser.add_argument("--scheduler_params", type=str, default="{}")
    parser.add_argument("--modalities", type=int, default=1)
    parser.add_argument("--bl_thr", type=float, default=0.0)
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--weights", type=str, default="", help="Stored weights to restore")
    parser.add_argument("--training_folders", type=str, nargs="+", default=["train"])
    parser.add_argument("--validation_folder", type=str, default="val")
    parser.add_argument(
        "--val_loader_id",
        type=int,
        default=-1,
        help="""
                            Kinda housefiry at the moment. When we have several train loader (for hybrid training
                            for instance), wants only one validation loader. The way the dataloading creation is
                            written at the moment, it will create several validation loader on the same topfolder (val),
                            but with different folders/bounds ; which will basically duplicate the evaluation.
                            """,
    )
    parser.add_argument("--augment", action="store_true") #when True, hardcoded to using the Augmentation fnc in dtaaloaders.py
    args = parser.parse_args()

    # args = parser.parse_args(['--dataset', 'minipaper/data_synt/', '--batch_size', '16', '--in_memory', '--l_rate', '0.001', '--schedule', \
    # 	'--n_epoch','30', '--workdir', 'minipaper/results/GTn_IN//orig_n_tmp','--csv', 'metrics.csv', '--n_class', '2',  '--modalities', '1', '--metric_axis', '0', '1', \
    # 	'--grp_regex', "(case_\d+_\d+)_\d+", '--network', 'ResidualUNet', '--losses', "[('GeneralizedDice', {'idc': [0,1]}, 1)]", '--folders', "[('IN', npy_transform, False), ('GT_noisy', gt_transform, True)] +[('GT_noisy', gt_transform, True)]"])

    if args.metric_axis == []:
        args.metric_axis = list(range(args.n_class))
    print("\n", args)

    return args


if __name__ == "__main__":
    run(get_args())
