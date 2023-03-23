#%%
#joining metrics over runs
from pathlib import Path
import pandas as pd
import numpy as np

import random 


def epread(fname):
    f = open(fname, 'r')
    ep = f.readline()
    f.close()
    return int(ep)

rootFolder = "results/acdc_geo_final_partialdata"
reps = 5
numclass = 4 #including bckg


#folders = [sorted(Path(rootFolder, f"rep{i}").glob("**")) for i in range(1, reps+1)]
folders = [sorted([child for child in Path(rootFolder, f"rep{i}").iterdir() if child.is_dir()]) for i in range(1, reps+1)]
assert len(set([len(i) for i in folders])) == 1, folders

count=0
debug=100
for allreps in zip(*folders):
    if count>debug: 
        break
    count+=1
    #print(allreps)
    assert len(set(a.name for a in allreps))==1, allreps
    assert len(allreps)==reps, (len(allreps), reps)
    name = allreps[0].name
    print("SUMMARIZING ", name)

    best_eps = [epread(Path(onerep, 'best_epoch.txt')) for onerep in allreps]
    print(f"BEST EPOCH\nmin {min(best_eps)}, avg {sum(best_eps)/reps}, max {max(best_eps)}")
    print()

    for metric in ['test_3d_dsc.npy', 'test_3d_hd95.npy']: #, 'test_dice.npy']:
        mets = np.stack([np.load(Path(onerep, metric)).squeeze() for onerep in allreps])
        assert mets.shape[-1]==numclass

        MeanOverSubsPerOrgan = mets.mean(axis=1) #shape reps x 50 x 4 -> reps x 4
        MeanOverSubsAll = mets.mean(axis=(1,2)) #MeanOverSubsPerOrgan.mean(axis=-1) #shape reps x 1
    #       MeanOverSubsForgr = MeanOverSubsPerOrgan[:,1:].mean(axis=-1) #shape reps x 1, only foreground classes

        ExperimentMean = mets.mean() #avg dice
        ExperimentStd = MeanOverSubsAll.std()

    #       ExperimentMeanForgr = MeanOverSubsForgr.mean() #avg freground dice
    #       ExperimentStdForgr = MeanOverSubsForgr.std() 
        
        ExperimentMeanPerOrgan = MeanOverSubsPerOrgan.mean(axis=0) #avg dice per class
        ExperimentStdPerOrgan = MeanOverSubsPerOrgan.std(axis=0)

        print(metric)
       # print(mets.mean(), mets.mean(axis=-1).std())
        print("PER ORGAN")
        print(ExperimentMeanPerOrgan, ExperimentStdPerOrgan)
        print("OVERALL")
        print(ExperimentMean, ExperimentStd)
        print("FOR PAPER:")
        print(ExperimentMeanPerOrgan[1:],ExperimentMean)
        print()
        print()







# %%
#plot distance maps
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import numpy as np
#%%

def nml(mp):
    mpnew = mp-mp.min()
    mpnew = mpnew/mpnew.max()
    return mpnew*255


p = 'data/ACDC-2D-GEO_fulldata/val'
subject = 'patient012_01_0_00'

img = np.asarray(Image.open(p+'/img/'+subject+'.png')).copy()
#img = nml(img)
gt = np.asarray(Image.open(p+'/random/'+subject+'.png')).copy()

img1, img2, img3 = img.copy(), img.copy(), img.copy()
img1[gt==1] = 0
img2[gt==2] = 0
img3[gt==3] = 0

gt1 = np.stack([img1, img1+(gt==1)*255, img1], axis=-1)
gt2 = np.stack([img2, img2+(gt==2)*255, img2], axis=-1)
gt3 = np.stack([img3, img3+(gt==3)*255, img3], axis=-1)

plt.figure(figsize=(10,10))
#plt.subplot(3,1,1)
plt.imshow(gt1)
plt.axis('off')
plt.figure(figsize=(10,10))
#plt.subplot(3,1,2)
plt.imshow(gt2)
plt.axis('off')
plt.figure(figsize=(10,10))
#plt.subplot(3,1,3)
plt.imshow(gt3)
plt.axis('off')

#%%
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

#%%
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

#%%
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


#%%

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
#plots of a few example predictions
# 
# 


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

#%%
enke = np.ones(gt.shape)
cl1 = np.stack([enke*255, enke*178, enke*102],axis=-1)
cl2 = np.stack([enke*153, enke*204, enke*255],axis=-1)
cl3 = np.stack([enke*110, enke*255, enke*0],axis=-1)
plt.imsave(f"class1.png", cl1.astype(np.uint8))
plt.imsave(f"class2.png", cl2.astype(np.uint8))
plt.imsave(f"class3.png", cl3.astype(np.uint8))


# %%
#making of a boxplot
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


sns.set(rc={'text.usetex' : True})


#font_scale=1.5, 
rep=1
n_class = 4
N = 50 #for poem: 10?
metr =  'hd95' #'dsc' #
folders2 = { 'ce': r"$\mathcal{L}_{\text{CE}}$, full annotation", 
           # 'ce_weak': 'pCE', #"$\\mathcal L_{\\widetilde{\\text{CE}}}$", 
            'ce_bl_eucl_point_fast': r"$\mathcal{L}_{\widetilde{\text{CE}}} + \mathcal{L}_B^{euc}$",
            'ce_bl_geo_point_fast': r"$\mathcal{L}_{\widetilde{\text{CE}}} + \mathcal{L}_B^{geo}$", 
            'ce_bl_int_point_fast': r"$\mathcal{L}_{\widetilde{\text{CE}}} + \mathcal{L}_B^{int}$", 
            'ce_bl_mbd_point': r"$\mathcal{L}_{\widetilde{\text{CE}}} + \mathcal{L}_B^{mbd}$"
            }
folders = { 'ce': 'CE', #"$\\mathcal L_{\\text{CE}}$, full annotation", 
            'ce_weak': 'pCE', #"$\\mathcal L_{\\widetilde{\\text{CE}}}$", 
            'ce_bl_eucl_point_fast': 'pCE + EUC', #$\\mathcal L_{\\widetilde{\\text{CE}}} + \\mathcal L_B^{euc}$",
            'ce_bl_geo_point_fast': 'pCE + GEO', #"$\\mathcal L_{\\widetilde{\\text{CE}}} + \\mathcal L_B^{geo}$", 
            'ce_bl_int_point_fast': 'pCE + INT', #"$\\mathcal L_{\\widetilde{\\text{CE}}} + \\mathcal L_B^{int}$", 
            'ce_bl_mbd_point': 'pCE + MBD' #"$\\mathcal L_{\\widetilde{\\text{CE}}} + \\mathcal L_B^{mbd}$"
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
#vpl.get_figure().savefig(f"ACDC{metr}.pdf", dpi=500)


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
metr = 'dsc'
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
plt.legend(bbox_to_anchor=(0.05, 1.15), loc='upper left', borderaxespad=0, ncol=3)
vpl.get_figure().savefig(f"ACDC{metr}_all.png", bbox_inches="tight", dpi=500)
plt.close(vpl.figure)

# %%
