import os
import sys
import glob
import scipy.io as sio
from PIL import Image, ImageFilter
from numpy import *
from scipy.misc import imread, imsave

annotation_dir = './trainval/trainval'#sys.argv[1] # path/to/annotation (without the last '/')
save_goal_dir = './Data/edge_image'

print 'annotation dir is: '+annotation_dir
print 'save edge image dir is: '+save_goal_dir


if not os.path.isdir(save_goal_dir):
    os.makedirs(save_goal_dir)


list_of_files = glob.glob(annotation_dir+'/*.mat') 
number = len(list_of_files)
count = 1
print 'finish load .mat file total number is %d' % number
for file_name in list_of_files:
    annotation = sio.loadmat(file_name)
    annotation_matrix = annotation['LabelMap']
    annotation_matrix = uint8(annotation_matrix)

    # print the length of the lines from the input file
    image = Image.fromarray(annotation_matrix,'RGB')
    image = image.resize((448,448,3));
    image = image.filter(ImageFilter.FIND_EDGES)
    image = image.convert('1')
    base = os.path.basename(file_name)
    base = os.path.splitext(base)[0]
    print 'number %d : save image %s' % (count, base)
    count = count+1
    image.save(save_goal_dir+'/'+base+'.png')

print 'finsih preparing edge image data and save successfully!!'



    
