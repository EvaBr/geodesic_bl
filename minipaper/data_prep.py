#script for running mini-paper experiments. 
#first, calculate varius Dts on images and their blurred/added noise counterparts
#(would it go on the fly??)
#learn UNET with dice+boundary loss using various DTS, to see what happens. 
#try with smoothed DTs (stochastic version)?
#try with weak annotations?

#UPDATE: weak annotations; as now, smoothed, maybe a bit translated etc.
#       but also: small(er) disks
#the raw data: not just 1/0, should be 1 (or close to) also some place outside the object
#
#Then: add some slices (axial?)  with liver, and try there. Or isles?
#%%
from functools import partial
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py 
from make_blobs import * #smooth out, random shift, generate image...
import matplotlib.pyplot as plt
import re 
from skimage.util import random_noise
from scipy.ndimage.morphology import distance_transform_edt as dte
from skimage.morphology import disk
from scipy import signal





#%% Create data
#LIVER
#cut patches with liver, and their dts
pot = '/tmp/.x2go-evabreznik/media/disk/_home_eva_Desktop_research_PROJEKT2-DeepLearning/AnatomyAwareDL/AnatomyNets/POEM_slices/'
outpath = 'data_liver'
thr = 10 #threshold: how many pixels should there at least be of the chosen organ, to keep the slice
organ = 3 #3 for liver, 4 for pancreas, etc...
create_organ_dataset(pot, outpath,thr,organ)


#%% CREATE SYNT
#parameters:
im_x, im_y = 256, 256
max_blobs_per_im = 3
max_bckg_blobs_im = 5
max_blob_radius = 90
blob_int_range = (0.6, 1)
bckg_blob_range = (0.3, 0.7) 
bckg_int_range = (0, 0.4)
nr_images = 1000
fld ="data_synt/"
create_data(fld, nr_images, im_x, im_y, max_blobs_per_im, max_bckg_blobs_im, blob_int_range, bckg_int_range, bckg_blob_range, max_blob_radius)
#create_data2()







#%% calc(or rather convert) DT files for liver data
fld = 'data_liver'
mape1 = ['lGT_IN','lGTn_IN', 'laGT_IN','laGTn_IN']
mape = ['TRAIN/'+x+'/GEO' for x in mape1] + ['TRAIN/'+x+'/MBD' for x in mape1] + ['VAL/'+x+'/GEO' for x in mape1] + ['VAL/'+x+'/MBD' for x in mape1]

probs = convert_mat_to_npy(fld, mape)
delete_problematic(fld, mape, probs)

#%% ...and for synt data:
tmppath = 'data_synt'
mape1 = ['GT_IN','GT_INn','GTn_IN','GTn_INn']
mape1 = [x+'/GEO' for x in mape1] + [x+'/MBD' for x in mape1]
mape = mape1 + ['a'+m for m in mape1]

probs = convert_mat_to_npy(tmppath, mape)
delete_problematic(tmppath, mape, probs)











#%%
#Sanity check, SYNTHETIC
# Visualize combo GT, IN, GT_IN, to see if they really belong together:
mp = 'val/'
fld = 'data_synt'
allnumbers = [re.findall(r'[0-9]+', p.name)[0] for p in Path(fld+mp+'GT_IN', 'GEO').glob('*.npy')]

#mp=''
GT = mp+'GT_noisy'
IN = mp+'IN'
COMBO = mp+'aGTn_IN'

koliko = 5
#files = np.random.choice(allnumbers, size=koliko, replace=False)
files = ['29', '295', '436', '825', '322']

for fil in range(koliko):
    namn = files[fil]
    gtpath = Path('data_synt', GT, 'im'+namn+'.npy')
    inpath = Path('data_synt', IN, 'im'+namn+'.npy')
    geopath = Path('data_synt', COMBO, 'GEO', 'im'+namn+'.npy')
    mbdpath = Path('data_synt', COMBO, 'MBD', 'im'+namn+'.npy')
    #readit = np.load
    #if geopath.suffix=='.mat':
    #   readit = lambda x: np.transpose(odpri_mat(x)[:,:,0])
        # OBS! MAT FILES ARE TRANSPOSED!
    gt, inn, geo, mbd = np.load(gtpath), np.load(inpath), np.load(geopath), np.load(mbdpath)

    plt.figure(figsize=(12,3.5))
    plt.suptitle(f'file nr {namn}')
    plt.subplot(1,4,1)
    plt.imshow(inn)
    plt.title('IN')
    plt.axis('off')
    plt.subplot(1,4,2)
    plt.imshow(gt)
    plt.title('GT')
    plt.axis('off')
    plt.subplot(1,4,3)
    plt.imshow(geo[:,:,0])
    plt.title('GEO')
    plt.axis('off')
    plt.subplot(1,4,4)
    plt.imshow(mbd[:,:,0])
    plt.title('MBD')
    plt.axis('off')





# %%
#sanity check, LIVER
mapca = 'val'
galim = 'GEO'

vsi = Path('data_liver', mapca, 'GT').glob('*.npy')
#en = np.random.choice(list(vsi))
en = list(Path('data_liver', mapca, 'GT').glob('*018_30_0.npy'))[0]
g = np.load(en)
i = np.load(f'data_liver/{mapca}/IN/{en.name}')
gn = np.load(f'data_liver/{mapca}/GT_noisy/{en.name}')

dta = np.load(f'data_liver/{mapca}/aGT_IN/{galim}/{en.name}')
dtan = np.load(f'data_liver/{mapca}/aGTn_IN/{galim}/{en.name}')
dt = np.load(f'data_liver/{mapca}/GT_IN/{galim}/{en.name}')
dtn = np.load(f'data_liver/{mapca}/GTn_IN/{galim}/{en.name}')


plt.figure()
plt.subplot(3,3,1)
plt.imshow(g)
plt.axis('off')
plt.subplot(3,3,2)
plt.imshow(gn)
plt.axis('off')
plt.subplot(3,3,3)
plt.imshow(i[0])
plt.axis('off')
plt.subplot(3,3, 4)
plt.imshow(dta[...,0])
plt.title('a'+galim)
plt.axis('off')
plt.subplot(3,3, 5)
plt.imshow(dtan[...,0])
plt.title('a'+galim+'_n')
plt.axis('off')
plt.subplot(3,3, 7)
plt.imshow(dt[...,0])
plt.title(galim)
plt.axis('off')
plt.subplot(3,3, 8)
plt.imshow(dtn[...,0])
plt.title(galim+'_n')
plt.axis('off')







#%% Functions for data creation and preparation

#helper to convert DTs (calc in matlab) to python and delete .mat files:
def odpri_mat(filpath):
    f = h5py.File(filpath, 'r')
    data = f.get('out')
    data = np.array(data)
    f.close()
    return data 


#SYNTHETIC data creation
def create_data2(fld, nr_images, im_x, im_y):
    #same as below except: now gaussian in foreground intensities. Then noise only superimposed. Also,sometimes foregeround intensities elsewhere than GT.
    kernel = np.outer(signal.windows.gaussian(70, 8),
                  signal.windows.gaussian(70, 8))
    return


def create_data(fld, nr_images, im_x, im_y, max_blobs_per_im, max_bckg_blobs_im, blob_int_range, bckg_int_range, bckg_blob_range, max_blob_radius):
    for img in tqdm(range(nr_images)):
        im, gt = generate_image(im_x, im_y, max_blobs_per_im, max_bckg_blobs_im, blob_int_range, 
                                bckg_int_range, bckg_blob_range, max_blob_radius)
        #create noisy labels. Can be randomly shifted, smoothed out, larger or smaller than original
        noise_type = [random_shift, smooth_out, blow_up]
    #    gt_noisy = np.random.choice(noise_type)(gt) #actually let's try with both always.
        gt_noisy = random_shift(smooth_out(gt))
        np.save(f"{fld}IN/im{img}.npy", im)
        np.save(f"{fld}GT/im{img}.npy", gt)
        np.save(f"{fld}GT_noisy/im{img}.npy", gt_noisy)

    #add noise to input im. Speckle noise atm (comes from imagin, may make sense?)
    for img in tqdm(range(nr_images)):
        im = np.load(f"{fld}IN/im{img}.npy")
        noisy = random_noise(im, mode='speckle', clip=True, mean=0., var=0.1)
        np.save(f"{fld}IN_noisy/im{img}.npy", noisy)
    
    return

# LIVER data creation
def create_organ_dataset(pot, outpath, threshold, organ):
    """pot...path to original (POEM) data
       outpath... where to save generated data
       threshold...min nr of pixels of chosen clas to keep the slice
       organ....which organ/class to keep"""

    for mp in ['TRAIN', 'VAL']:
        gtpaths = Path(pot, mp, 'gt').glob('*.npy')
        inpath = Path(pot, mp, 'in1')
    
        gtout = Path(outpath, mp, 'gt')
        gtoutN = Path(outpath, mp, 'gt_noisy') #'noisy', degenerated GT
        inout = Path(outpath, mp, 'in')
        gtout.mkdir(parents=True, exist_ok=True)
        gtoutN.mkdir(parents=True, exist_ok=True)
        inout.mkdir(parents=True, exist_ok=True)

        for gtp in gtpaths:
            gt = np.load(gtp)
            gt = gt[organ, ...].squeeze()
            if gt.sum()>threshold: #keep the slice 
                deggt = degenerate_gt1(gt)
                #check if ok? <- should be. Even for m//2=0, disk nonempty in degenerate_gt!
                np.save(Path(gtoutN, gtp.name), deggt)
                np.save(Path(gtout, gtp.name), gt)
                infil = np.load(Path(inpath, gtp.name))
                np.save(Path(inout, gtp.name), infil[0:2,...]) #DT channels are not needed
            

    with open('info.txt', 'w') as f:
        f.write(f'Slices of organ {organ}. Threshold to keep slice {threshold}.')
        f.close()
    return


# calc(or rather convert) DT files from mat to npy
def convert_mat_to_npy(filespath, mape):
    problemdict = {mapa:[] for mapa in mape}
    for folder in mape: 
        print(folder)
        Path(filespath, folder).mkdir(parents=True, exist_ok=True)
        for filpath in Path(filespath, folder).glob("dt*"):
            pid = re.findall()
            try:
                #filpath = Path(filespath, folder, f'dt{nr}.mat') #f'{folder}/dt{nr}.mat'
                fil = np.transpose(odpri_mat(filpath), axes=(1,0,2))#%%
                np.save(Path(fld, folder, filpath.stem).with_suffix(".npy"), fil)
                filpath.unlink() #delete the .mat file
            # print(nr)
            except OSError:
                problemdict[folder].append(pid)
        #if len(problemi)>0:
        #    print(f'Problemi v fajlih: {problemi}\n')
    return problemdict


#ni druge, kr odstranimo vse te problematicne fajle.
def delete_problematic(topfolder, mape, problems):
    base_mape = ['GT', 'GT_noisy', 'IN', 'IN_noisy']#%%
    for mapa in mape:
        for filnr in problems:
            fajl = list(Path(topfolder, mapa).glob(f'*{filnr}.npy'))[0]
            fajl.unlink(missing_ok=True)



def degenerate_gt1(gt):
    #keeps the centre of GT, makes a disk around it. Probably makes sense for organ data
    newgt = np.zeros(gt.shape)

    dt = dte(gt) 
    m = dt.max()
    m_xy = np.where(dt == m )
    center_X, center_Y = m_xy[0][0], m_xy[1][0]

    diskgt = disk(m//2)
    s = diskgt.shape[0]

    newgt[center_X-s//2 : center_X-s//2+s, center_Y-s//2: center_Y-s//2+s] = diskgt
    return newgt


#separate train/val, to work with orig. code
#Works only w folders in data_synt! (LIVER already separated originally)
def separate_train_val(nrval, nrtest, mapall, topfolder="data_synt"):
    allnumbers = [int(re.findall(r'[0-9]+', p.name)[0]) for p in Path('data_synt', 'GT_IN', 'GEO').glob('*.npy')]

    val = np.random.choice(allnumbers, size=nrval+nrtest, replace=False)
    test = np.random.choice(val, size=nrtest, replace=False)
    train = [v for v in allnumbers if v not in val]
    val = [v for v in val if v not in test]


    for mapa in mapall:
        train_pot = Path(topfolder, 'train', mapa)
        train_pot.mkdir(parents=True, exist_ok=True)
        Path(topfolder, mapa).replace(train_pot)
    
        val_pot = Path(topfolder, 'val', mapa)
        val_pot.mkdir(parents=True, exist_ok=True)

        test_pot = Path(topfolder, 'test', mapa)
        test_pot.mkdir(parents=True, exist_ok=True)
        

        for fil in val:
            fajl = list(train_pot.glob(f'*[m|t]{fil}.npy'))[0]
            fajl.rename(Path(val_pot, fajl.name))
    
        for fil in test:
            fajl = list(train_pot.glob(f'*[m|t]{fil}.npy'))[0]
            fajl.rename(Path(test_pot, fajl.name))

        Path(topfolder, mapa).rmdir() #remove empty dir
    #after this, removed (by hand) empty folders... unless above line is called. 

