import numpy as np
import argparse
import re
import pandas as pd
from PIL import Image
from pathlib import Path
from utils import probs2one_hot, probs2class
from utils import dice_coef, dice_batch
from scipy.ndimage import distance_transform_edt as eucl_dist

def finalEval(args):
    #_, val_loaders = get_loaders(args, args.dataset,
    #                        args.batch_size, args.n_class,
    #                        args.debug, args.in_memory, args.dimensions, args.use_spacing)

    #no need to comlicate life with using this. simply go through all pairs of 
    #  (outpputfolder/final_epoch/im*.png, outputfolder/best_epoch/im*.png, datasetfolder/GT/im*.png) 
    # and load those and eval. 
    #then write to csv

    folders = eval(args.folders)
    gtpath = list(Path(args.dataset).glob('*.npy'))
    L = len(gtpath)
    allbckg = np.zeros((L,1))
    allforg = np.zeros((L,1))
    outbckgfinal = np.zeros((L,1))
    outforgfinal = np.zeros((L,1))
    truebckgfinal = np.zeros((L,1))
    trueforgfinal = np.zeros((L,1))
    outbckgbest = np.zeros((L,1))
    outforgbest = np.zeros((L,1))
    truebckgbest = np.zeros((L,1))
    trueforgbest = np.zeros((L,1))
    ahdbest = np.zeros((L,1))
    ahdfinal = np.zeros((L,1))   
    imageid = np.zeros((L,1))
    
    nrepoch = args.nrepochs

    for outs in folders:
        finalpath = Path( "results",  outs, 'iter'+str(nrepoch).zfill(3), 'val')
        bestpath = Path( outs, 'best_epoch', 'val')
        for i,GT in enumerate(gtpath):
            gt = np.load(GT)
            getid = re.findall("[0-9]+.*[0-9]+", GT.name)
            imageid[i] = getid
            final = np.array(Image.open(Path(finalpath, f'im{getid}.png')))
            best = np.array(Image.open(Path(bestpath, f'im{getid}.png')))
            #need to calc AHD here since we need eucliddts
            gteuc = eucl_dist(gt)
            finaleuc = eucl_dist(final)
            besteuc = eucl_dist(best)

            numel = np.prod(gt.shape)
            allforg[i] = sum(gt)
            allbckg[i] = numel - allforg[i]
            outforgfinal[i] = sum(final)
            outbckgfinal[i] = numel - outforgfinal[i]
            outforgbest[i] = sum(best)
            outbckgbest[i] = numel - outforgbest[i]
            trueforgfinal[i] = sum(gt*final)
            trueforgbest[i] = sum(gt*best)
            truebckgfinal[i] = sum(gt==final)-trueforgfinal[i]
            truebckgbest[i] = sum(gt==best)-trueforgbest[i]
            #AHD per image <- we can average, since bcs all imgs are
            #  same size, we can get overall average by simply additionally
            #  dividing their sum by nr of imges
            ahdfinal[i] = np.mean(np.logical_xor(final, gt)*(finaleuc**2+gteuc**2))
            ahdbest[i] = np.mean(np.logical_xor(best, gt)*(besteuc**2+gteuc**2))
            

        df = pd.DataFrame((imageid, allbckg, allforg, outbckgfinal, outforgfinal, outbckgbest, \
            outforgbest, truebckgfinal, trueforgfinal, truebckgbest, trueforgbest, ahdfinal, ahdbest), 
            columns="""imageid allbckg allforg outbckgfinal outforgfinal outbckgbest outforgbest 
            truebckgfinal trueforgfinal truebckgbest trueforgbest ahdfinal ahdbest""".split())
        df.set_index('imageid')
        df.to_csv(Path(args.savedir, args.csv))

    calcmetrics(args) #stupid but whatevs, searatedfor better modularity

def calcmetrics(args):
    data = pd.read_csv(Path(args.savedir, args.csv))
    resdict = {}
    #dices
    resdict['avg_person_f_dice'] = np.mean((2*data['trueforgfinal']+0.001)/(data['allforg']+data['outforgfinal']+0.001))
    resdict['avg_person_b_dice'] = np.mean((2*data['trueforgbest']+0.001)/(data['allforg']+data['outforgbest']+0.001))
    resdict['dataset_f_dice'] = 2*sum(data['trueforgfinal'])/sum(data['allforg']+data['outforgfinal']) #no rgularization necessary, we now there's at least one img with foreground in dataset
    resdict['dataset_b_dice'] = 2*sum(data['trueforgbest'])/sum(data['allforg']+data['outforgbest'])

    #recall
    resdict['avg_person_f_rec'] = np.mean((data['trueforgfinal']+0.001)/(data['allforg']+0.001))
    resdict['avg_person_b_rec'] = np.mean((data['trueforgbest']+0.001)/(data['allforg']+0.001))
    resdict['dataset_f_rec'] = sum(data['trueforgfinal'])/sum(data['allforg'])
    resdict['dataset_b_rec'] = sum(data['trueforgbest'])/sum(data['allforg'])
    #precision
    resdict['avg_person_f_pre'] = np.mean((data['trueforgfinal']+0.001)/(data['outforgfinal']+0.001))
    resdict['avg_person_b_pre'] = np.mean((data['trueforgbest']+0.001)/(data['outforgbest']+0.001))
    resdict['dataset_f_pre'] = (sum(data['trueforgfinal'])+0.001)/(sum(data['outforgfinal'])+0.001)
    resdict['dataset_b_pre'] = (sum(data['trueforgbest'])+0.001)/(sum(data['outforgbest'])+0.001)
    
    #AHD
    resdict['avg_person_ahd_f'] = np.mean(data['ahdfinal'])
    resdict['avg_person_ahd_b'] = np.mean(data['ahdbest'])

    
    with open(Path(args.savedir, args.csv).with_suffix('txt'), 'w') as f:
        f.writelines([f"{namn}\n{value}\n" for namn,value in resdict.items])
    print("\n".join([f"{namn}:  {value}" for namn, value in resdict.items]))






def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Hyperparams')
        parser.add_argument('--dataset', type=str, required=True) #folder of GT. eg. 'data_synt/val/GT
        parser.add_argument("--csv", type=str, required=True) #where to save results
     #   parser.add_argument("--losses", type=str, required=True) #which metrics to calc. Dice etc
        parser.add_argument("--savedir", type=str, required=True) #folder inside results to save to.

        #parser.add_argument("--debug", action="store_true")
        parser.add_argument("--cpu", action='store_true') #
        parser.add_argument("--folders", type=str, required=True,
                            help = "List of outputfolders.")
        parser.add_argument("--nrepochs", type=int, required=True)
        args = parser.parse_args()

        return args


if __name__ == '__main__':
        finalEval(get_args())

      