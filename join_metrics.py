#%%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import random 


def epread(fname):
    f = open(fname, 'r')
    ep = f.readline()
    f.close()
    return int(ep)


def nml(mp):
    mpnew = mp-mp.min()
    mpnew = mpnew/mpnew.max()
    return mpnew*255




#%%
######  joining metrics over runs, to compute stats

###  ACDC  ###
#rootFolder = "results/acdc_geo_final_partialdata"
#numclass = 4 #including bckg

###  POEM  ###
rootFolder = "results/poem_last_2D"
numclass = 7 #including bckg


reps = 5


np.set_printoptions(precision=3, suppress=True)
folders = [sorted([child for child in Path(rootFolder, f"rep{i}").iterdir() if child.is_dir()]) for i in range(1, reps+1)]
assert len(set([len(i) for i in folders])) == 1, folders

count=0
debug=10000 #set to 5 or small number if debugging
for allreps in zip(*folders):
    if count>debug: 
        break
    count+=1

    assert len(set(a.name for a in allreps))==1, allreps
    assert len(allreps)==reps, (len(allreps), reps)
    name = allreps[0].name
  #  print("SUMMARIZING ", name)

    best_eps = [epread(Path(onerep, 'best_epoch.txt')) for onerep in allreps]
  #  print(f"BEST EPOCH\nmin {min(best_eps)}, avg {sum(best_eps)/reps}, max {max(best_eps)}")
  #  print()

    #print(name)
    sezki = {'dsc': None, 'hd95': None}
    for metric in ['test_3d_dsc.npy', 'test_3d_hd95.npy', 'test_dice.npy']:
        whichone = metric.split("_")[-1]
        whichone = whichone.split(".")[0]
        mets = np.stack([np.load(Path(onerep, metric)).squeeze() for onerep in allreps])
        assert mets.shape[-1]==numclass

        MeanOverSubsPerOrgan = mets.mean(axis=1) #shape reps x nrSubjects x nrOrgans -> reps x nrOrgans
        MeanOverSubsAll = mets.mean(axis=(1,2)) #shape reps x 1
    #       MeanOverSubsForgr = MeanOverSubsPerOrgan[:,1:].mean(axis=-1) #shape reps x 1, only foreground classes

        ExperimentMean = mets.mean() #avg dice
        ExperimentStd = MeanOverSubsAll.std()

    #       ExperimentMeanForgr = MeanOverSubsForgr.mean() #avg freground dice
    #       ExperimentStdForgr = MeanOverSubsForgr.std() 
        
        ExperimentMeanPerOrgan = MeanOverSubsPerOrgan.mean(axis=0) #avg dice per class
        ExperimentStdPerOrgan = MeanOverSubsPerOrgan.std(axis=0)

      #  print(metric)
      #  print("PER ORGAN")
      #  print(ExperimentMeanPerOrgan, ExperimentStdPerOrgan)
      #  print("OVERALL")
      #  print(ExperimentMean, ExperimentStd)
      #  print("FOR PAPER:")
      #  print( " & ".join([f"${a:.03f} (\pm{b:.03f})$" for a,b in zip(list(ExperimentMeanPerOrgan[1:])+[ExperimentMean], list(ExperimentStdPerOrgan[1:])+[ExperimentStd])]) )
       # print( whichone.upper()+ " " +" & ".join([f"${a:.03f} $" for a in list(ExperimentMeanPerOrgan[1:])+[ExperimentMean]]) + "\\")
     #   print()
      #  print()
        sezki[whichone] = list(ExperimentMeanPerOrgan[1:])+[ExperimentMean]
    #now jon them
    print(name + " " + "&".join([f" $\\uparrow{a:.03f}\\downarrow{b:.03f}$ " for a,b in zip(sezki['dsc'], sezki['hd95'])]))





# %%
#######  plot distance maps

p = 'data/ACDC-2D-GEO_fulldata/val'
subject = 'patient012_01_0_00'

img = np.asarray(Image.open(p+'/img/'+subject+'.png')).copy()
#img = nml(img)
gt = np.asarray(Image.open(p+'/random/'+subject+'.png')).copy()



mbd = np.load(p+'/mbd_point/'+subject+'.npy')
#mbd = nml(mbd)
mbd1 = mbd[1] #Image.open(p+'/mbd_point_1/'+subject+'.png')
mbd2 = mbd[2] #Image.open(p+'/mbd_point_2/'+subject+'.png')
mbd3 = mbd[3] #Image.open(p+'/mbd_point_3/'+subject+'.png')

plt.figure(figsize=(10,10))
plt.imshow(mbd1)
plt.axis('off')
plt.figure(figsize=(10,10))
plt.imshow(mbd2)
plt.axis('off')
plt.figure(figsize=(10,10))
plt.imshow(mbd3)
plt.axis('off')


geo = np.load(p+'/geo_point_fast/'+subject+'.npy')
geo = nml(geo)
geo1 = geo[1] #Image.open(p+'/geo_point_fast_1/'+subject+'.png')
geo2 = geo[2] #Image.open(p+'/geo_point_fast_2/'+subject+'.png')
geo3 = geo[3] #Image.open(p+'/geo_point_fast_3/'+subject+'.png')

plt.figure(figsize=(10,10))
plt.imshow(geo1)
plt.axis('off')
plt.figure(figsize=(10,10))
plt.imshow(geo2)
plt.axis('off')
plt.figure(figsize=(10,10))
plt.imshow(geo3)
plt.axis('off')


inte = np.load(p+'/int_point_fast/'+subject+'.npy')
#inte=nml(inte)
int1 = inte[1] #Image.open(p+'/int_point_fast_1/'+subject+'.png')
int2 = inte[2] #Image.open(p+'/int_point_fast_2/'+subject+'.png')
int3 = inte[3] #Image.open(p+'/int_point_fast_3/'+subject+'.png')
plt.figure(figsize=(10,10))
plt.imshow(int1)
plt.axis('off')
plt.figure(figsize=(10,10))
plt.imshow(int2)
plt.axis('off')
plt.figure(figsize=(10,10))
plt.imshow(int3)
plt.axis('off')


euc = np.load(p+'/eucl_point_fast/'+subject+'.npy')
#euc = nml(euc)
euc1 = euc[1] #Image.open(p+'/eucl_point_fast_1/'+subject+'.png')
euc2 = euc[2] #Image.open(p+'/eucl_point_fast_2/'+subject+'.png')
euc3 = euc[3] #Image.open(p+'/eucl_point_fast_3/'+subject+'.png')
plt.figure(figsize=(10,10))
plt.imshow(euc1)
plt.axis('off')
plt.figure(figsize=(10,10))
plt.imshow(euc2)
plt.axis('off')
plt.figure(figsize=(10,10))
plt.imshow(euc3)
plt.axis('off')


# %%
##############    plots of a few example predictions

vsi = list(Path('data/ACDC-2D-GEO_fulldata/test/img').glob('*.png'))
subjects = random.choices(vsi, k=10)
path = Path('results/acdc_geo_final/rep1')
folders = ['ce', 'ce_weak', 'ce_bl_eucl_point_fast','ce_bl_geo_point_fast', 'ce_bl_int_point_fast', 'ce_bl_mbd_point']
folderNames = ['ce', 'ce_weak', 'euc', 'geo', 'int', 'mbd']
for subj in subjects:
    img = np.asarray(Image.open(f'data/ACDC-2D-GEO_fulldata/test/img/{subj.name}'))
    gt = np.asarray(Image.open(f'data/ACDC-2D-GEO_fulldata/test/gt/{subj.name}'))
    
    ss = subj.with_suffix('').name
    print(ss)

    #SHOW THE IMG+GT
    gt1, gt2, gt3 = gt==1, gt==2, gt==3
    imggt = img*(gt==0)
    imggt = np.stack([imggt+gt1*255+gt2*153+gt3*110, imggt+gt1*178+gt2*204+gt3*255, imggt+gt1*102+gt2*255+gt3*0], axis=-1)
    plt.figure(figsize=(10,10))
    plt.imshow(imggt)
    plt.axis('off')
    plt.imsave(f"FullGT_{ss}.png", imggt.astype(np.uint8))
    print('GT')

    for folder, folderName in zip(folders, folderNames):
        out = np.asarray(Image.open(str(Path(path, folder, 'iter000/test', subj.name)))).astype(np.uint8)
        #save img+out

        imggt = img*(out==0)
        gt1, gt2, gt3 = out==1, out==2, out==3

        imggt = np.stack([imggt+gt1*255+gt2*153+gt3*110, imggt+gt1*178+gt2*204+gt3*255, imggt+gt1*102+gt2*255+gt3*0], axis=-1)
        plt.figure(figsize=(10,10))
        plt.imshow(imggt)
        plt.axis('off')
        plt.imsave(f"{folderName}_{ss}.png", imggt.astype(np.uint8))
        print(folderName)

#now save also squares of class colours
enke = np.ones(gt.shape)
cl1 = np.stack([enke*255, enke*178, enke*102],axis=-1)
cl2 = np.stack([enke*153, enke*204, enke*255],axis=-1)
cl3 = np.stack([enke*110, enke*255, enke*0],axis=-1)
plt.imsave(f"class1.png", cl1.astype(np.uint8))
plt.imsave(f"class2.png", cl2.astype(np.uint8))
plt.imsave(f"class3.png", cl3.astype(np.uint8))


# %%
###############    making of a boxplot

sns.set(rc={'text.usetex' : True})
#plt.rcParams['text.usetex'] = True
 
rep=1
n_class = 4
N = 50 #for poem: 10
metr =  'hd95' #'dsc' #'dice'
folders = { 'ce': r'$\mathcal{L}_{\text{CE}}$, full annotation', 
            'ce_weak': r'$\mathcal L_{\widetilde{\text{CE}}}$', 
            'ce_bl_eucl_point_fast': r'$\mathcal{L}_{\widetilde{\text{CE}}} + \mathcal{L}_B^{euc}$',
            'ce_bl_geo_point_fast': r'$\mathcal{L}_{\widetilde{\text{CE}}} + \mathcal{L}_B^{geo}$', 
            'ce_bl_int_point_fast': r'$\mathcal{L}_{\widetilde{\text{CE}}} + \mathcal{L}_B^{int}$', 
            'ce_bl_mbd_point': r'$\mathcal{L}_{\widetilde{\text{CE}}} + \mathcal{L}_B^{mbd}$',
            'ce_rloss': r'$\mathcal{L}_{\widetilde{\text{CE}}} + $CRF-loss'
            }


#%%
C = np.concatenate(
    [np.load(
        f'results/acdc_geo_final_partialdata/rep{rep}/{foldr}/test_3d_{metr}.npy'
        ).squeeze() 
    for foldr in folders]
    )
#add average
#C = np.concatenate([C,C.mean(axis=1)[:,np.newaxis]], axis=1)
dttypes = [val for val in folders.values() for i in range(N)]
plotData = pd.DataFrame({'Mean 3D '+metr.upper():C.mean(axis=1), 'Training setting': dttypes})

plt.rcParams['text.usetex'] = True
vpl = sns.boxplot(data=plotData, x="Training setting", y='Mean 3D '+metr.upper())
vpl.get_figure().savefig(f"ACDC{metr}.pdf", dpi=500)
plt.close(vpl.figure)

#%%
metr = 'dsc' #
#per class
for foldr in folders:
    C = np.load(f'results/acdc_geo_final_partialdata/rep{rep}/{foldr}/test_3d_{metr}.npy').squeeze()
    SS = pd.DataFrame(C[:, 1:])
    S = pd.concat([SS.iloc[:,i] for i in range(n_class-1)], ignore_index=True)
    #classes = ['BLD']*N*Nf + ['KDL']*N*Nf + ['LVR']*N*Nf + ['PNC']*N*Nf+['SPL']*N*Nf+['KDR']*N*Nf
    classes = ['LV']*N + ['Myo']*N + ['RV']*N
    nameData = pd.DataFrame({'Class': classes})

    skupaj = pd.concat([S, nameData], axis=1)
    skupaj.rename(columns={0:'3D '+metr.upper()}, inplace=True)


    vpl = sns.boxplot(data=skupaj, x="Class", y='3D '+metr.upper())
    vpl.get_figure().savefig(f"ACDC{metr}_{foldr}.png")
    plt.close(vpl.figure)

# %%

#folders = {i:j for i,j in folders.items() if i!="ce_weak"}
metr = 'hd95'
Nf = len(folders)
C = np.concatenate([np.load(f'results/acdc_geo_final_partialdata/rep{rep}/{foldr}/test_3d_{metr}.npy').squeeze() for foldr in folders])
SS = pd.DataFrame(C[:, 1:])
S = pd.concat([SS.iloc[:,i] for i in range(n_class-1)], ignore_index=True)
#classes = ['BLD']*N*Nf + ['KDL']*N*Nf + ['LVR']*N*Nf + ['PNC']*N*Nf+['SPL']*N*Nf+['KDR']*N*Nf
classes = ['LV']*N*Nf + ['Myo']*N*Nf + ['RV']*N*Nf
methods = [nm for nm in folders.values() for i in range(N)]*(n_class-1)
nameData = pd.DataFrame({'Class': classes, 'Training setting': methods})

skupaj = pd.concat([S, nameData], axis=1)
skupaj.rename(columns={0:'3D '+metr.upper()}, inplace=True)


vpl = sns.boxplot(data=skupaj, x="Class", y='3D '+metr.upper(), hue="Training setting")
plt.legend(bbox_to_anchor=(0.02, 1.2), loc='upper left', borderaxespad=0, ncol=3)
vpl.get_figure().savefig(f"ACDC{metr}_all.png", bbox_inches="tight", dpi=500)
plt.close(vpl.figure)

# %%
