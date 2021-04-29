#%%
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
import random
from PIL import Image 
import matplotlib.patches as mpatches


#%%
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
        gtfilepath = Path(gt_path, filenr)
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
    plt.subplot(1,2,2)
    im=plt.imshow(out, vmin=0, vmax=7)
    plt.title('OUT')
    plt.axis('off')

    values = np.arange(7)
    colors = [ im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label=organs[i]) for i in range(len(values)) ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1.), loc=2, borderaxespad=0. )

    plt.show()

#%%
results_folder = 'poem'
out_folder1 = 'gdl2' 
#epoch = 99
epoch = None
filenr = 'case_500022_0_35.png'
#filenr = 'case_500022_0_71.png'
#filenr = 'case_500242_0_38.png'
#filenr = 'case_500348_0_44.png'
compareOuts(results_folder, out_folder1, filenr=filenr, epoch=epoch)


out_folder2 = 'gdl2_w' 
compareOuts(results_folder, out_folder2, filenr=filenr, epoch=epoch)
# %%
