import argparse
import warnings
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from dataloader import SliceDataset, png_transform_nonorm,  tensor_transform, dummy_gt_transform, custom_collate
from utils import save_images, map_, tqdm_, probs2class


def runInference(args: argparse.Namespace):
        print('>>> Loading model')
        net = torch.load(args.model_weights)
        device = torch.device("cuda")
        net.to(device)

        print('>>> Loading the data')
        batch_size: int = args.batch_size
        num_classes: int = args.num_classes

        folders: list[Path] = [Path(args.data_folder)]

        imtype: str = "npy" if args.data_folder[-3:]=="npy" else "png"  #args.imgtype
        if imtype=="png":
                imtrans = png_transform_nonorm
        else: #npy
                imtrans =  tensor_transform
        names: list[str] = map_(lambda p: str(p.name), folders[0].glob(f"*.{imtype}"))
        #print(folders, len(names), names[:10])
        dt_set = SliceDataset(names,
                              folders * 2,  # Duplicate for compatibility reasons
                              are_hots=[False, False],
                              transforms=[imtrans, dummy_gt_transform],  # So it is happy about the target size
                              debug=args.debug,
                              K=num_classes,
                              ignore_norm=True)
        loader = DataLoader(dt_set,
                            batch_size=batch_size,
                            num_workers=batch_size + 2,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=custom_collate)

        print('>>> Starting the inference')
        savedir: Path = Path(args.save_folder)
        savedir.mkdir(parents=True, exist_ok=True)
        total_iteration = len(loader)
        desc = ">> Inference"
        tq_iter = tqdm_(enumerate(loader), total=total_iteration, desc=desc)
        with torch.no_grad():
                for j, data in tq_iter:
                        filenames: list[str] = data["filenames"]
                        image: Tensor = data["images"].to(device)

                        pred_logits: Tensor = net(image)
                        pred_probs: Tensor = F.softmax(pred_logits, dim=1)

                        with warnings.catch_warnings():
                                warnings.simplefilter("ignore")

                                predicted_class: Tensor
                                predicted_class = probs2class(pred_probs)
                                save_images(predicted_class, filenames, savedir, mode="test", iter=0)


def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Inference parameters')
        parser.add_argument('--data_folder', type=str, required=True, 
                            help="The folder containing the images to predict")
        parser.add_argument('--save_folder', type=str, required=True)
        parser.add_argument('--model_weights', type=str, required=True)

        parser.add_argument("--debug", action="store_true")
     #   parser.add_argument("--imgtype", type=str, default="png", choices=["png", "npy"],
     #                           help="What type will the image input be (in case of POEM, 2 channel data, png is not suitable).")

        parser.add_argument('--num_classes', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=10)
        parser.add_argument("--group", action="store_true",
            help="Group the patient slices together for validation. \
                Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.",
        )
        parser.add_argument("--grp_regex", type=str, default=None)
        #parser.add_argument("--csv", type=str, required=True)

        args = parser.parse_args()

        print(args)

        return args


if __name__ == '__main__':
        args = get_args()
        runInference(args)