#%%
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
import random
from PIL import Image 
import matplotlib.patches as mpatches

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

#%%
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

def compareOuts(results_folder, out_folder, epoch=None, gt_folder='gt', filenr=None):
    gt_path = Path('data/POEM/val/', gt_folder)
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
                compareOuts(results_folder, out_folder, epoch, gt_folder, filenr=gfp.name)
            return 
        else: #we only plot 1 img 
            gtfilepath = gtfilepath[0]
            filenr = gtfilepath.name
    outfilepath = Path(out_path, filenr)

    #read and plot both files:
    gt = Image.open(gtfilepath).convert('L')
    out = Image.open(outfilepath).convert('L')

    iternr = outfilepath.parent.parent
    subtitle = f"({iternr.parent.name}, ep{iternr.name[-3:]})"

    organs = ['Bckg', 'Bladder', 'KidneyL', 'Liver', 'Pancreas', 'Spleen', 'KidneyR']
    plt.figure(figsize=(12, 5))
    plt.suptitle(f"{filenr[5:-4]}\n{subtitle}")
    plt.subplot(1,2,1)
    plt.imshow(gt, vmin=0, vmax=7)
    plt.title('GT')
    plt.axis('off')
    ax2 = plt.subplot(1,2,2)
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
    t = ax2.text(1.08, 0.5, 'Dices:', size='medium', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    for d in range(7): 
        t = ax2.text(1.085, 0.45-d*0.05, f"{organs[d]}: {dajci[d]:.3f}", size='small', transform=ax2.transAxes)

    plt.show()

#%%
results_folder = 'poemAlla'
out_folder1 = 'gdl' 
epoch = 30
#epoch = None
filenr = 'case_500022_0_35.png'
#filenr = 'case_500022_0_71.png'
#filenr = 'case_500242_0_38.png'
#filenr = 'case_500348_0_44.png'
compareOuts(results_folder, out_folder1, gt_folder='gt', filenr=filenr, epoch=epoch)


results_folder = 'poem'
out_folder2 = 'unw_opt2_tmp' 
compareOuts(results_folder, out_folder2, gt_folder='gt', filenr=filenr, epoch=epoch)
# %%
