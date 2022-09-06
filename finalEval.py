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
    gtpath = list(Path("minipaper", args.dataset).glob('*.npy'))
    L = len(gtpath)
    allbckg = np.zeros((L,))
    allforg = np.zeros((L,))
    outbckgfinal = np.zeros((L,))
    outforgfinal = np.zeros((L,))
    truebckgfinal = np.zeros((L,))
    trueforgfinal = np.zeros((L,))
    outbckgbest = np.zeros((L,))
    outforgbest = np.zeros((L,))
    truebckgbest = np.zeros((L,))
    trueforgbest = np.zeros((L,))
    ahdbest = np.zeros((L,))
    ahdfinal = np.zeros((L,))   
    imageid = []
    
    nrepoch = args.nrepochs

    for outs in folders:
        print(outs)
        imageid = []
        finalpath = Path( "minipaper/results",  args.savedir, outs, 'iter'+str(nrepoch-1).zfill(3), 'val')
        bestpath = Path( "minipaper/results",  args.savedir, outs, 'best_epoch', 'val')
        for i,GT in enumerate(gtpath):
            gt = np.load(GT)
           # print(GT.name)
            getid = re.findall("[0-9]+", GT.name)
            getid = "_".join(getid)
           # print(getid)
            imageid.append(getid)
            final = np.array(Image.open(Path(finalpath, f'im{getid}.png')))
            best = np.array(Image.open(Path(bestpath, f'im{getid}.png')))
            #need to calc AHD here since we need eucliddts
            gteuc = eucl_dist(gt)
            finaleuc = eucl_dist(final)
            besteuc = eucl_dist(best)

            numel = np.prod(gt.shape)
            allforg[i] = gt.sum()
            allbckg[i] = numel - allforg[i]
            outforgfinal[i] = final.sum()
            outbckgfinal[i] = numel - outforgfinal[i]
            outforgbest[i] = best.sum()
            outbckgbest[i] = numel - outforgbest[i]
            trueforgfinal[i] = np.sum(gt*final)
            trueforgbest[i] = np.sum(gt*best)
            truebckgfinal[i] = np.sum(gt==final)-trueforgfinal[i]
            truebckgbest[i] = np.sum(gt==best)-trueforgbest[i]
            #AHD per image <- we can average, since bcs all imgs are
            #  same size, we can get overall average by simply additionally
            #  dividing their sum by nr of imges
            ahdfinal[i] = np.mean(np.logical_xor(final, gt)*(finaleuc**2+gteuc**2))
            ahdbest[i] = np.mean(np.logical_xor(best, gt)*(besteuc**2+gteuc**2))
            

        
        df = pd.DataFrame({"allbckg":allbckg, "allforg":allforg, "outbckgfinal":outbckgfinal, "outforgfinal":outforgfinal, "outbckgbest":outbckgbest, \
            "outforgbest":outforgbest, "truebckgfinal":truebckgfinal, "trueforgfinal":trueforgfinal, "truebckgbest":truebckgbest, "trueforgbest":trueforgbest,\
                "ahdfinal": ahdfinal, "ahdbest":ahdbest}, index=imageid)
        
        df.to_csv(Path("minipaper/results", args.savedir, outs, args.csv))
        
        calcmetrics(df, Path("minipaper/results", args.savedir, outs, args.csv)) #stupid but whatevs, searatedfor better modularity
        


def calcmetrics(data, savepath):
    #data = pd.read_csv(Path("results", args.savedir,  args.csv))
   
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

    
    with open(savepath.with_suffix('.txt'), 'w') as f:
        f.writelines([f"{namn}\n{value}\n" for namn,value in resdict.items()])
    print("\n".join([f"{namn}:  {value}" for namn, value in resdict.items()]))






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
        #args = parser.parse_args()
        args = parser.parse_args(["--nrepochs", '30',  "--folders", "['orig', 'orig_n', 'mbd', 'geo', 'ambd', 'ageo', 'euc']", "--savedir", "SYNT", "--csv",
                "eval.csv", "--dataset", "data_synt/val/GT"])

        return args


if __name__ == '__main__':
        finalEval(get_args())

      