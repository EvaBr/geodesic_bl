import numpy as np
import argparse
import re
from PIL import Image
from pathlib import Path
from utils import probs2one_hot, probs2class
from utils import dice_coef, dice_batch

def finalEval(args):
    #_, val_loaders = get_loaders(args, args.dataset,
    #                        args.batch_size, args.n_class,
    #                        args.debug, args.in_memory, args.dimensions, args.use_spacing)

    #no need to comlicate life with using this. simply go through all pairs of 
    #  (outpputfolder/final_epoch/im*.png, outputfolder/best_epoch/im*.png, datasetfolder/GT/im*.png) 
    # and load those and eval. 
    #then write to csv

    folders = eval(args.folders)
    gtpath = Path(args.dataset)
    for outs in folders:
        nrepoch = outs[1]
        finalpath = Path('results', outs[0], 'iter'+str(nrepoch).zfill(3), 'val')
        bestpath = Path('results', outs[0], 'best_epoch', 'val')
        for GT in gtpath.glob('*.npy'):
            gt = np.load(GT)
            getid = re.findall("[0-9]+.*[0-9]+", GT.name)
            final = np.array(Image.open(Path(finalpath, f'im{getid}.png')))
            best = np.array(Image.open(Path(bestpath, f'im{getid}.png')))

            


def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Hyperparams')
        parser.add_argument('--dataset', type=str, required=True) #folder of GT. eg. 'adata_synt/val/GT
        parser.add_argument("--csv", type=str, required=True) #where to save results
        parser.add_argument("--losses", type=str, required=True) #which metrics to calc. Dice etc
        parser.add_argument("--n_class", type=int, required=True)

        #parser.add_argument("--debug", action="store_true")
        parser.add_argument("--cpu", action='store_true') #
        #parser.add_argument("--in_memory", action='store_true')
        #parser.add_argument("--use_spacing", action='store_true')
        #parser.add_argument("--no_assert_dataloader", action='store_true')
        #parser.add_argument("--ignore_norm_dataloader", action='store_true')
       # parser.add_argument("--group", action='store_true') 
        #parser.add_argument("--group_train", action='store_true')#false for 2D
        #parser.add_argument("--grp_regex", type=str, default=None) #only used for grouping
        #parser.add_argument("--modalities", type=int, default=1)
        #parser.add_argument("--dimensions", type=int, default=2)
        #parser.add_argument('--batch_size', type=int, default=1)
        #parser.add_argument("--training_folders", type=str, nargs="+", default=["train"])
        #parser.add_argument("--validation_folder", type=str, default="val")
        #parser.add_argument("--val_loader_id", type=int, default=-1)
        parser.add_argument("--folders", type=str, required=True,
                            #help="List of list of (subfolder, transform, is_hot)")
                            help = "List of (outputfolder, nroflastepoch).")
        args = parser.parse_args()

        return args


if __name__ == '__main__':
        finalEval(get_args())

      