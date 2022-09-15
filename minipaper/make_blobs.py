#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from skimage import morphology
from pathlib import Path
import matplotlib.pyplot as plt


#%%
def random_shift(gt, amount=None):
    smeri = np.random.choice([[0], [1], [2], [3], [0,3], [0,2], [1,3], [1,2]]) #L, R, D, U, LU, LD, RU, RD
    if amount==None:
        amount = np.random.choice(np.arange(5,25)) #cca 2-10%
    newgt = np.zeros(gt.shape)
    s0,s1 = gt.shape
    for smer in smeri:
        newgt[(smer==1)*amount:s0-(smer==0)*amount, (smer==2)*amount:s1-(smer==3)*amount] = gt[amount*(smer==0):s0-amount*(smer==1), amount*(smer==3):s1-amount*(smer==2)] #move L/R, U/D
    return newgt

def smooth_out(gt, amount=None):
    #actually, let's just do closing + one more erosion
    if amount==None:
        amount = np.random.choice(np.arange(30, 50)) #10-30
    tmp = morphology.binary_closing(gt, morphology.square(amount) )#disk(amount))

    return morphology.binary_erosion(tmp, morphology.disk(15)) #1

def blow_up(gt, amount=None):
    #actually, let's just do opening+one more dilation
    if amount==None:
        amount = np.random.choice(np.arange(10, 40))
    tmp = morphology.binary_opening(gt, morphology.disk(amount))
    return morphology.binary_dilation(tmp, morphology.disk(1))


#%%
def generate_image(x, y, max_s, max_bs, s_range, b_range, bs_range, max_rad): #already as created a bit "noisy"
    newim = np.zeros((x,y), dtype=np.float)
    newgt = np.zeros((x,y), dtype=np.uint8)
    newbckg = np.zeros((x,y), dtype=np.uint8)

    actual_blobs = np.random.randint(1, max_s+1) #assume we want AT LEAST 1
    bckg_blobs = np.random.randint(1, max_bs+1)
    for s in range(bckg_blobs):
        center = np.random.randint(0, [x, y])
        #now create a small polygon around this center
        polygon = create_polygon(center, x, y, max_rad) #gives you tuple of pixel indexes belonging to blob
        newim[polygon] = np.random.uniform(bs_range[0], bs_range[1], len(polygon[0]))
        newbckg[polygon] = 1
        
    for s in range(actual_blobs):
        center = np.random.randint(0, [x, y])
        #now create a small polygon around this center
        polygon = create_polygon(center, x, y, max_rad) #gives you tuple of pixel indexes belonging to blob
        newim[polygon] = np.random.uniform(s_range[0], s_range[1], len(polygon[0]))
        newgt[polygon] = 1
        newbckg[polygon] = 1
    
    #add also bckg data as appropriate intensities:
    newim[newbckg==0] = np.random.uniform(b_range[0], b_range[1], np.sum(newbckg==0))

    return newim, newgt

    

#%%
def create_polygon(center, x, y, max_radius=25, step=30):
    #walk randomly for a while?
    #get actual walk limits:
   # x_0, y_0 = np.maximum(center-max_radius, [0, 0])
   # x_1, y_1 = np.minimum(center+max_radius, [x, y])
    #random walker aparently takes a while. Instead, lets do 12 angles, choose 
    #radius at each angle, get corresponding pixel. between pixels, join with a line?
    points = [[center[0]], [center[1]]]
    for angle in range(0, 360, step):
        degs = 2*np.pi*angle/360.
        ro = np.random.rand()*max_radius
        #get actual pixel this reaches to:
        pix = [np.ceil(center[0]+ro*np.cos(degs)), np.ceil(center[1]+ro*np.sin(degs))]
        pix = np.maximum([0, 0], np.minimum(pix, [x-1, y-1]))
        points[0].append(int(pix[0]))
        points[1].append(int(pix[1]))
    

    #now connect the points:
    N = len(points[0])
    xs, ys = [], []
    for pt in range(N):
        ptt = (pt+1)%N
        x0, y0, x1, y1 = points[0][pt], points[1][pt], points[0][ptt], points[1][ptt]
        
    #    print((x0,y0, x1, y1))

        if x0==x1 and y0==y1:
            continue
    

        transpose = abs(x1 - x0) < abs(y1 - y0)
        if transpose:
            x0, y0, x1, y1 = y0, x0, y1, x1
        if x0 > x1: # Swap line direction to go left-to-right if necessary
            x0, y0, x1, y1 = x1, y1, x0, y0
        # Compute intermediate coordinates using line equation
        xi = np.arange(x0 + 1, x1)
        yi = np.floor(((y1 - y0) / (x1 - x0)) * (xi - x0) + y0).astype(xi.dtype)
        # Write intermediate coordinates
        if transpose:
            xi, yi = yi, xi
    #    if not (np.all(xi<x) and np.all(yi<y)):
    #        print((xi, yi))
        xs.extend(xi) 
        ys.extend(yi)
        
    points[0].extend(xs)
    points[1].extend(ys)

    #now we have a polygon boundary. We also want to fill it. 
    filing1 = fill_polygon(points)

    #to do the filind also by rows:
    filing2 = fill_polygon([points[1], points[0]])

    #add this to orig. points
    points[0].extend(filing1[0])
    points[1].extend(filing1[1])

    points[0].extend(filing2[1])
    points[1].extend(filing2[0])

    return tuple(points)

def fill_polygon(pointsi):
    #scan bottom-up, fill each line individually:
    points = [[],[]]
    filing = [[],[]]
    miny, maxy = min(pointsi[1]), max(pointsi[1])
    #first make a shallow copy, ignoring max and min elements
    for ptx, pty in zip(*pointsi):
        if pty>miny and pty<maxy:
            points[0].append(ptx)
            points[1].append(pty)
    #now get new min/max:
    miny, maxy = min(points[1]), max(points[1])

    for row in range(miny, maxy+1):
        cols = [p for idx,p in enumerate(points[0]) if points[1][idx]==row]
        cols.sort()
        for c in range(1): #(0, len(cols)-1, 2):
            tofill = [i for i in range(cols[0], cols[-1])] #(cols[c]+1, cols[c+1])]
            filing[0].extend(tofill)
            filing[1].extend([row]*len(tofill))
        #for now we just fill everything between the first and last occurence 
        #    of a certain row/column. This way polygons get more 'smoothed out', 
        #   but it doesnt matter/might even be good for this simple dataset.

    return filing
# %%
def plot_example():
    allim = sorted(Path('IN').glob('*'))
    allimn = sorted(Path('IN_noisy').glob('*'))
    allgt = sorted(Path('GT').glob('*'))
    allgtn = sorted(Path('GT_noisy').glob('*'))
    k = np.random.randint(0,len(allim))

    img = np.load(allim[k])
    imgn = np.load(allimn[k])
    gt = np.load(allgt[k])
    gtn = np.load(allgtn[k])

    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(img)    
    plt.subplot(1,4,2)
    plt.imshow(imgn)    
    plt.subplot(1,4,3)
    plt.imshow(gt)    
    plt.subplot(1,4,4)
    plt.imshow(gtn)
# %%
