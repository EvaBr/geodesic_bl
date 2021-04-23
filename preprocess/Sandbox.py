#%%
import pickle 
import re 
import glob

#let's post-cutting save the poem 3d spacing dictionary, which is now needed. 
dest_path = f"../data/POEM"
resolution_dict = {}
for mode in ['train', 'val', 'test']:
    pids = [re.findall(r"500[0-9]+", fil)[0] for fil in glob.glob(f"{dest_path}/in_npy/*")]
    resolution_dict |= {pid: (2.070313, 2.070313, 8.0) for pid in pids}

with open(f"{dest_path}/spacing_3d.pkl", 'wb') as f:
    pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saved spacing dictionary to {f}")
# %%
