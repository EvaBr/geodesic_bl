#%%
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
import random
from PIL import Image 
import matplotlib.patches as mpatches
import pandas as pd
from glob import glob

def check2Dcuts(pid):
    findit = glob(f"./data/POEM/*/gt/*{pid}.png")
    findit2 = glob(f"./data/POEM/*/gt_pts/*{pid}.png")
    gt = Image.open(findit[0]).convert('L')
    gtp = Image.open(findit2[0]).convert('L')
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(gt)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(gtp)
    plt.axis('off')
   
    plt.show()

def get_one_hot(gt, nb_class):
    gt = gt.astype('int')
    classes = np.eye(nb_class)
    one_hot = classes[gt]
    s = np.arange(one_hot.ndim)
    return np.transpose(one_hot, (s[-1],*s[:-1]))

def AllDices(probs, target):
    pc = get_one_hot(probs, 7).astype(np.float32)
    tc = get_one_hot(target, 7).astype(np.float32)
    
    intersection = np.einsum("cwh,cwh->c", pc, tc)
    union = (np.einsum("cwh->c", pc) + np.einsum("cwh->c", tc))

    divided = (2 * intersection + 1e-10) / (union + 1e-10)
    #if class present neither in GT nor OUT in the whole batch, the output==1

    return divided

def compareOuts(results_folder, out_folder, epoch=None, plot_pts=False, filenr=None):
    gt_folder='gt'
    gt_pts_folder = 'gt_pts'
    gt_path = Path('data/POEM/val/', gt_folder)
    gt_pts_path = Path('data/POEM/val/', gt_pts_folder)
    out_path = Path('results', results_folder, out_folder, 'best_epoch/val')
    if epoch!=None:
        #don't use best epoch
        out_path = Path('results', results_folder, out_folder, 'iter'+str(epoch).zfill(3), 'val')

    if filenr==None:
        #choose a file at random
        gtfilepath = random.choice(list(gt_path.glob('*.png')))
        filenr = gtfilepath.name
    else:
        #let's find the given pid. could be in train or val folder
        #TODO: IF IT's IN TRAIN, png DOES NOT EXIST! NEED TO LOAD NETWORK AND EVAL. 
        gtfilepath = list(Path('data/POEM').glob(f'*/{gt_folder}/*{filenr}*'))
        if len(gtfilepath)>1: #we wish to plot multiple times
            for gfp in gtfilepath:
                compareOuts(results_folder, out_folder, epoch, plot_pts=plot_pts, filenr=gfp.name)
            return 
        else: #we only plot 1 img 
            gtfilepath = gtfilepath[0]
            filenr = gtfilepath.name
    outfilepath = Path(out_path, filenr)
    gtptsfilepath = Path(gt_pts_path, filenr)

    #read and plot both files:
    gt = Image.open(gtfilepath).convert('L')
    gtpts = Image.open(gtptsfilepath).convert('L')
    out = Image.open(outfilepath).convert('L')

    iternr = outfilepath.parent.parent
    subtitle = f"({iternr.parent.name}, ep{iternr.name[-3:]})"

    organs = ['Bckg', 'Bladder', 'KidneyL', 'Liver', 'Pancreas', 'Spleen', 'KidneyR']
    inrow=2
    if plot_pts:
        inrow=3
    plt.figure(figsize=(inrow*5+2, 6))
    plt.suptitle(f"{filenr[5:-4]}\n{subtitle}")
    plt.subplot(1,inrow,1)
    plt.imshow(gt, vmin=0, vmax=7)
    plt.title('GT')
    plt.axis('off')
    if plot_pts:
        plt.subplot(1,inrow,2)
        plt.imshow(gtpts, vmin=0, vmax=7)
        plt.title('GT PTS')
        plt.axis('off')
    ax2 = plt.subplot(1,inrow,inrow)
    im = plt.imshow(out, vmin=0, vmax=7)
    plt.title('OUT')
    plt.axis('off')

    values = np.arange(7)
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label=organs[i]) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1.), loc=2, borderaxespad=0. )

    dajci = AllDices(np.asarray(out), np.asarray(gt))
    dajcipts = AllDices(np.asarray(out), np.asarray(gtpts))
    t = ax2.text(1.08, 0.5, 'Dices:', size='medium', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    for d in range(7): 
        t = ax2.text(1.085, 0.45-d*0.05, f"{organs[d]}: {dajci[d]:.3f}"+ (f"({dajcipts[d]:.3f})" if plot_pts else ""), size='small', transform=ax2.transAxes)

    plt.show()

#%%
def compareOutsGeneral(results_folder, out_folders, gt_folder, other_gt=None, epoch=None, filenr=None):
    """results_folder = main results folder (results, minipaper/results).
        out_folders = [GT_IN/euc, poem/gdl], or one folder up in the tree, dependin on results_folder.
        gt_folder = data/POEM/val/gt, or minipaper/data_synt/val/GT.
        epoch = epoch nr to plot. If None. best epoch.
        filenr = im004 etc, no suffix.
        other_gt = string path again, to real/degenrated, another GT. Eg. minipper/data_sy/v/GT_noisy """
    gt_path = Path(gt_folder)
    epochstr = 'best_epoch'
    if epoch!=None:
        #don't use best epoch
        epochstr = 'iter'+str(epoch).zfill(3)

    out_paths = [Path(results_folder, i, epochstr, 'val') for i in out_folders]
    
    if filenr==None:
        #choose a file at random
        gtfilepath = random.choice(list(gt_path.glob('*.npy')))
        filenr = gtfilepath.stem
    else:
        #let's find the given pid. could be in train or val folder
        gtfilepath = list(gt_path.glob(f'*{filenr}.npy'))
        if len(gtfilepath)>1: #we wish to plot multiple times
            for gfp in gtfilepath:
                compareOutsGeneral(results_folder, out_folders, gt_folder, epoch, filenr=gfp.stem)
            return 
        else: #we only plot 1 img 
            gtfilepath = gtfilepath[0]
            filenr = gtfilepath.stem
    outfilepaths = [list(outp.glob(filenr+'*'))[0] for outp in out_paths]
    print(filenr)
    #read and plot both files:
    gt = np.load(gtfilepath)
    #print(gt.shape)
    out = [Image.open(outfile).convert('L') for outfile in outfilepaths]
    if isinstance(other_gt, str):
        gt2 = list(Path(other_gt).glob(f'*{filenr}.npy'))
        gt2 = np.load(gt2[0])
    
    classes = ['Background'] + [f'Object {i+1}' for i in range(np.max(gt))]
    inrow=2+isinstance(other_gt,str)

    L = len(out_folders)
    plt.figure(figsize=(inrow*5+2, 4*L))
    plt.suptitle(filenr + f', epoch {epoch}')
    plt.subplot(L,inrow,1)
    plt.imshow(gt, vmin=0, vmax=len(classes))
    plt.title('GT')
    plt.axis('off')
    if inrow==3:
        plt.subplot(L,inrow,2)
        plt.imshow(gt2, vmin=0, vmax=len(classes))
        plt.title('GT2')
        plt.axis('off')
    for idx,i in enumerate(out_folders):
        plt.subplot(L,inrow,inrow*(idx+1))
        im = plt.imshow(out[idx], vmin=0, vmax=len(classes))
        plt.title('OUT, ' + i)
        plt.axis('off')

    values = np.arange(len(classes))
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label=classes[i]) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1.), loc=2, borderaxespad=0. )
    plt.show()

    #for i in out:
    #    dajci = AllDices(np.asarray(i), np.asarray(gt))
    #    print(dajci)
    #    if inrow==3:
    #        dajci2 = AllDices(np.asarray(i), np.asarray(gt2))
    #        print(dajci2)
#%%   
compareOutsGeneral('minipaper/results/GTn_INp/p0.8', ['orig','euc','mbd','geo'],'minipaper/data_synt/val/GT', 'minipaper/data_synt/val/GT_noisy', filenr='im317')
compareOutsGeneral('minipaper/results/GTn_INp/p0.2', ['orig','euc','mbd','geo'],'minipaper/data_synt/val/GT', 'minipaper/data_synt/val/GT_noisy', filenr='im317')

#%%
def compare_curves(resultfolder, list_of_names, plot_names = None, individ_Dices = [0,1,2,3,4,5,6]):
    if plot_names==None:
        plot_names = list_of_names
    #read in metrics
    metrics = {plotname: pd.read_csv(f"results/{resultfolder}/{name}/metrics.csv") for plotname,name in zip(plot_names, list_of_names)}
    Dice_names = ['Dice_bck','Dice_Bladder', 'Dice_KidneyL', 'Dice_Liver', 'Dice_Pancreas', 'Dice_Spleen', 'Dice_KidneyR']
    to_plot = [Dice_names[i] for i in individ_Dices]
    csv_names = [f'Dice{i}' for i in individ_Dices]

    L = len(to_plot)+1
    cols, rows = min(3,L), np.ceil(L/3)
    for tip in [("tra_","TRAINING"), ("val_","VALIDATION")]:
        plt.figure(figsize = (cols*7,rows*5))
        plt.suptitle(tip[1])
        plt.subplot(rows, cols, 1)
        plt.title('Total Loss')
        for name in metrics:
            totalLoss = metrics[name].filter(regex=f"{tip[0]}Loss")
            plt.plot(totalLoss.sum(axis=1), label=name)
        plt.legend()

        for idx, what in enumerate(csv_names):
            plt.subplot(rows, cols, idx+2)
            plt.title(to_plot[idx])
            for name in metrics:
                plt.plot(getattr(metrics[name], f"{tip[0]}{what}"), label=name)
            plt.legend()
        plt.show()
    
#%%
compare_curves('poemBaseline', ['gdl_1', 'gdl_c', 'gdl_w_c', 'gdl_w_1'])

#%%
results_folder = 'poemBaseline'
out_folder1 = 'gdl_w_1' 
epoch = 1
#epoch = None
filenr = 'case_500022_0_35.png'
filenr = 'case_500022_0_71.png'
filenr = 'case_500242_0_38.png'
filenr = 'case_500348_0_44.png'
compareOuts(results_folder, out_folder1, plot_pts=True, filenr=filenr, epoch=epoch)

out_folder2 = 'gdl_w_c' 
compareOuts(results_folder, out_folder2, plot_pts=True, filenr=filenr, epoch=epoch)
# %%
