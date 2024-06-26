#!/usr/bin/env python3.9

import argparse
from pathlib import Path

import numpy as np


def main(args) -> None:
    print(f"Reporting on {len(args.folders)} folders.")

    main_metric: str = args.metrics[0]

    best_epoch: list[int] = display_metric(args, main_metric, args.folders, args.axises)
    for metric in args.metrics[1:]:
        display_metric(args, metric, args.folders, args.axises, best_epoch)


def display_metric(args, metric: str, folders: list[str], axises: tuple[int], best_epoch: list[int] = None):
    print(f"{metric} (classes {axises})")

    if not best_epoch:
        get_epoch = True
        best_epoch = [0] * len(folders)
    else:
        get_epoch = False

    for i, folder in enumerate(folders):
        file: Path = Path(folder, metric).with_suffix(".npy")
        data: np.ndarray = np.load(file)
        if data.ndim==2: #quickfix for being able to use it with test eval, where epoch dim is trivial
            data = data[np.newaxis, ...]
        data = data[:, :, axises]  # Epoch, sample, classes
        averages: np.ndarray = data.mean(axis=(1, 2))
        stds: np.ndarray = data.std(axis=(1, 2))

        class_wise_avg: np.ndarray = data.mean(axis=1)
        class_wise_std: np.ndarray = data.std(axis=1)

        if get_epoch:
            if args.mode == "max":
                best_epoch[i] = np.argmax(averages)
            elif args.mode == "min":
                best_epoch[i] = np.argmin(averages)

        val: float
        val_std: float
        if args.mode in ['max', 'min']:
            val = averages[best_epoch[i]]
            val_std = stds[best_epoch[i]]
            val_class_wise = class_wise_avg[best_epoch[i]]
        else:
            val = averages[-args.last_n_epc:].mean()
            val_std = averages[-args.last_n_epc:].std()
            val_class_wise = class_wise_avg[-args.last_n_epc:].mean(axis=0)

        assert val_class_wise.shape == (len(axises),)

        precision: int = args.precision
        print(f"\t{Path(folder).name}: {val:.{precision}f} ({val_std:.{precision}f}) at epoch {best_epoch[i]}")
        if len(axises) > 1 and args.detail_axises:
            val_cw_std = class_wise_std[best_epoch[i]]
            assert val_cw_std.shape == (len(axises),)

            # print(f"\t\t {' '.join(f'{a}={val_class_wise[j]:.{precision}f}' for j,a in enumerate(axises))}")
            print(f"\t\t {' '.join(f'{a}={val_class_wise[j]:.{precision}f} ({val_cw_std[j]:.{precision}f})' for j,a in enumerate(axises))}")

    return best_epoch


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot data over time')
    parser.add_argument('--folders', type=str, required=True, nargs='+', help="The folders containing the file")
    parser.add_argument('--metrics', type=str, required=True, nargs='+')
    parser.add_argument('--axises', type=int, required=True, nargs='+')
    parser.add_argument('--mode', type=str, default='max', choices=['max', 'min', 'avg'])
    parser.add_argument('--last_n_epc', type=int, default=1)
    parser.add_argument('--precision', type=int, default=4)
    parser.add_argument('--debug', action='store_true', help="Dummy for compatibility.")

    parser.add_argument('--detail_axises', action='store_true',
                        help="Print each axis value on top of the mean")

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
