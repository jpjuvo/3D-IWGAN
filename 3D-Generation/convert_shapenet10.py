import numpy as np
import sys
import os
import scipy.io
import glob
#from path import Path
from tqdm import tqdm

if sys.argv[-1] == '-v': # this will allow you to visualize the models as they are made, more of a sanity check 
    import mayavi.mlab
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from mpl_toolkits.mplot3d import Axes3D


instances = {}
class_id_to_name = {
    "1": "cup",
    "2": "monitor",
    "3": "chair",
    "4": "desk",
    "5": "car",
    "6": "bowl",
    "7": "vase",
    "8": "lamp",
    "9": "table",
    "10": "person"
}
class_name_to_id = { v : k for k, v in class_id_to_name.items() }
class_names = set(class_id_to_name.values())


if not os.path.exists('data/train/'):
    os.makedirs('data/train/')
if not os.path.exists('data/test/'):
    os.makedirs('data/test/')
base_dir = (sys.argv[2])

for (dirpath, dirnames, filenames) in tqdm(os.walk(base_dir)):
    for filename in filenames:
        if filename.endswith('.mat'):
            fname = os.path.join(dirpath,filename)
            if fname.endswith('test_feature.mat') or fname.endswith('train_feature.mat'): 
                continue
            elts = os.path.split(fname) #fname.splitall()
            nameonly = os.path.splitext(elts[1])
            info = nameonly[0].split('_')
            if len(info)<3: continue  
            if info[0] == 'discriminative' or info[0] == 'generative' : continue 
            instance = info[1]
            rot = int(info[2])
            dirs = dirpath.split('\\')					
            split = dirs[len(dirs) - 1]
            classname = dirs[len(dirs) - 3].strip()
            if classname in class_names:
                dest = 'data/'+split+'/' + classname + '/'
            else:
                continue
            if not os.path.exists(dest):
                os.makedirs(dest)
            arr = scipy.io.loadmat(fname)['instance'].astype(np.uint8)
            matrix = np.zeros((32,)*3, dtype=np.uint8)
            matrix[1:-1,1:-1,1:-1] = arr
            if sys.argv[-1] == '-v':
                xx, yy, zz = np.where(matrix>= 0.3)
                mayavi.mlab.points3d(xx, yy, zz,
                                 
                                     color=(.1, 0, 1),
                                     scale_factor=1)

                mayavi.mlab.show()
            # saves the models by instance name, and then rotation 
            np.save(dest +  instance + '_' + str(rot) , matrix)
        else:
            continue
 


