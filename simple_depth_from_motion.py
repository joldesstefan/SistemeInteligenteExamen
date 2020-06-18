#!/usr/bin/env python
# coding: utf-8

# In[26]:



import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
#from image_warping import warp_image






import tensorflow as tf
#import tf_lie

import tensorflow as tf

def add_two_trailing_dims(x):
   return tf.expand_dims(tf.expand_dims(x, axis=-1), axis=-1)

def transpose_matrix_collection(x):
   axes = list(range(len(x.get_shape())))
   target_axes = axes[:-2] + list(reversed(axes[-2:]))
   return tf.transpose(x, perm=target_axes)

def dim2TransView(u, w):
   T1 = tf.constant([[0,0,0,1],
                     [0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,0]])
   
   T2 = tf.constant([[0,0,0,0],
                     [0,0,0,1],
                     [0,0,0,0],
                     [0,0,0,0]])
   
   T3 = tf.constant([[0,0,0,0],
                     [0,0,0,0],
                     [0,0,0,1],
                     [0,0,0,0]])
   
   T4 = tf.constant([[0,0, 0,0],
                     [0,0,-1,0],
                     [0,1, 0,0],
                     [0,0, 0,0]])
   
   T5 = tf.constant([[0,0,-1,0],
                     [0,0, 0,0],
                     [1,0, 0,0],
                     [0,0, 0,0]])
   
   T6 = tf.constant([[0,-1, 0,0],
                     [1, 0, 0,0],
                     [0, 0, 0,0],
                     [0, 0, 0,0]])
   translations = tf.cast(tf.stack([T1, T2, T3]), tf.float32)
   rotations =    tf.cast(tf.stack([T4, T5, T6]), tf.float32)
   return tf.reduce_sum(add_two_trailing_dims(u)*translations
                       + add_two_trailing_dims(w)*rotations, axis=-3)


def dim1TransView(w):
   T1 = tf.constant([[0,0, 0],
                     [0,0,-1],
                     [0,1, 0]])
   
   T2 = tf.constant([[0,0,-1],
                     [0,0, 0],
                     [1,0, 0]])
   
   T3 = tf.constant([[0,-1, 0],
                     [1, 0, 0],
                     [0, 0, 0]])
   rotations = tf.cast(tf.stack([T1, T2, T3]), tf.float32)
   return tf.reduce_sum(add_two_trailing_dims(w)*rotations, axis=-3)


def gammaThC(R): # log
   theta = tf.acos((tf.trace(R)-1)/2)
   return add_two_trailing_dims(theta / (2 * tf.sin(theta))) * (R - transpose_matrix_collection(R))

def egalizeGen(R):
   wx = gammaThC(R)
   gamma = tf.acos((tf.trace(R)-1)/2)
   A = add_two_trailing_dims(tf.sin(gamma) / gamma)
   B = add_two_trailing_dims((1 - tf.cos(gamma)) / (gamma ** 2))
   C = (1 - A) / (add_two_trailing_dims(gamma) ** 2)
   I = tf.eye(3)
   V = I + B*wx + C*tf.matmul(wx, wx)
   return V

def globEgApy(C): # log
   
   R = C[...,:3,:3]
   t = C[...,:3, 3]
   wx = gammaThC(R)

   V = egalizeGen(R)
   Vinv = tf.linalg.inv(V)
   u = tf.matmul(Vinv, tf.expand_dims(t, axis=-1))
   empty_row = tf.zeros(V.shape[:-2].as_list() + [1,4]) 
   WXu = tf.concat([wx, u], axis=-1)
   result = tf.concat([WXu, empty_row], axis=-2)
   return result
   


def finalViewT(u, w): # exp
   wx = dim1TransView(w)
   gamma = tf.sqrt(tf.reduce_sum(w * w, axis=-1))
   A = add_two_trailing_dims(tf.sin(gamma) / (gamma))
   B = add_two_trailing_dims((1 - tf.cos(gamma)) / ((gamma ** 2)))
   C = (1 - A) / (add_two_trailing_dims(gamma) ** 2)
   I = tf.eye(3)
   R = I + A*wx + B*tf.matmul(wx, wx)
   V = I + B*wx + C*tf.matmul(wx, wx)
   Vu = tf.matmul(V, tf.expand_dims(u, axis=-1))
   empty_4x4 = tf.zeros(Vu.shape[:-2].as_list() + [1,4]) 
   row_0001 = empty_4x4 + tf.eye(num_rows=1, num_columns=4)[...,::-1]
   RVu = tf.concat([R, Vu], axis=-1)
   result = tf.concat([RVu, row_0001], axis=-2)
   return result





def imgTotTransform(pcI, depth, u, w):
   # equation 3 in engel 2014
   # https://vision.in.tum.de/_media/spezial/bib/engel14eccv.pdf
   l = [pcI[...,1] / depth,
        pcI[...,0] / depth,
        1.0 / depth,
        tf.ones_like(depth)]
   pc = tf.expand_dims((tf.stack(l, axis=-1)), axis=-1)
   tr = finalViewT(u, w)

   # blank_transform_map is an all zeros tensor if shape
   # [batch size, height, width, 4, 4] that will contain the 4x4
   # homogenous transform for each pixel
   blank_transform_map = tf.zeros(pc.shape[:3].as_list() + [4,4])

   # change the transform matrix batch's shape so we can broadcast
   # it over the spatial dimensions of the image batch
   tr = tf.reshape(tr, [-1,1,1,4,4])
   ctm = tr + blank_transform_map
   
   warped_pixel_location = tf.matmul(ctm, pc)[...,0]
   return tf.stack([warped_pixel_location[...,0], 
                    warped_pixel_location[...,1],
                    tf.ones_like(depth_image)], axis=-1) \
                       / tf.expand_dims(warped_pixel_location[...,2], axis=-1)

def distorsion(image, depth, u, w):
   
   n_rows =  image.shape.as_list()[1]
   n_cols = image.shape.as_list()[2]
   rows = tf.range(0.0, n_rows, 1.0) / n_rows
   cols = tf.range(0.0, n_cols, 1.0) / n_cols
   coords = tf.stack(tf.meshgrid(cols, rows), axis=-1)
   
   distorsion_normalized_pixel_coords = warp(coords, depth, u, w)[...,:2]
   distorsion_pixel_coords  = distorsion_normalized_pixel_coords * tf.Variable([n_rows * 1.0,
                                                                        n_cols * 1.0])
   
   distorsion_image = tf.contrib.resampler.resampler(
       image_batch,
       distorsion_pixel_coords[...,::-1])
   
   convFil = tf.cast(warped_pixel_coords[...,0] > 0, tf.float32) *                       tf.cast(warped_pixel_coords[...,0] < n_rows-1, tf.float32) *                       tf.cast(warped_pixel_coords[...,1] > 0, tf.float32)*                       tf.cast(warped_pixel_coords[...,1] < n_cols-1, tf.float32)
   distorsion_image *= tf.expand_dims(convFil, axis=-1)
   return distorsion_image


# # Load images

# In[18]:


image_paths = [
"data/24.jpg",
"data/18.jpg",
"data/21.jpg",
"data/27.jpg",
"data/33.jpg",
"data/36.jpg"]
crop_row = 50
crop_col = 290
importImgs = [mpimg.imread(path)[crop_row:crop_row+400,crop_col:crop_col + 400] for path in image_paths]

images = [im / 255.0 for im in importImgs ]
for im in images:
    plt.imshow(im)
    plt.show()


# # Define optimization problem

# In[29]:


tf.reset_default_graph()

originalImage = tf.cast(tf.constant(images[0]), tf.float32)
# create a batch dimension
originalImage = tf.expand_dims(originalImage, axis=0)

# ensure depth value is positive
depth =  tf.abs(tf.Variable(tf.ones(originalImage[...,0].shape))) +0.2

toLo = tf.image.total_variation((tf.expand_dims((depth), axis=-1))) / (400**2) * 0.2

cost = toLo
distorImgs = []
for scene_image in images[1:]:
    scene_image = tf.cast(tf.constant(scene_image), tf.float32)
    scene_image = tf.expand_dims(scene_image, axis=0)

    # pose representation for the cameras that took each scene image
    translation = tf.Variable([[0.01, 0.01, 0.01]])
    rotation = tf.Variable([[0.01, 0.01, 0.01]])

    distorImg = warp_image(scene_image, depth,
                                    translation, rotation)

    cost += tf.losses.huber_loss(distorImg, originalImage)
    distorImgs.append(distorImg)
    
optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(cost)

init = tf.global_variables_initializer()

# A version of Tensorflow released since I wrote the original post
# caused a regression where NaNs are produced in the output depth
# map after one iteration when running on the GPU.  So that users
# don't hit this by default, this runs on CPU.

config = tf.ConfigProto(
        # Run CPU only
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)
sess.run(init)
cost_value_history = []


# # Run optimization

# In[ ]:


#make figures larger
import matplotlib
matplotlib.rcParams['figure.figsize'] = [7, 7]
import cv2

def turn_off_tick_marks():
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,   
        left=False,
        labelbottom=False,
        labelleft=False) 
    
def save_figure(filename):
    plt.savefig(filename, dpi=250, facecolor='#eee8d5', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=True, bbox_inches='tight', pad_inches=0.0,
        frameon=None, metadata=None)

n_steps = 5000
image_index = 0

plt.title("the reference image")
plt.imshow(sess.run(originalImage)[0])
turn_off_tick_marks()
plt.show()

for i in range(n_steps+1):
    _, cost_val = sess.run([optimizer,cost])
    cost_value_history.append(cost_val)

    if i==4999 and i>0:
        print ("iteration", i)
        plt.plot(cost_value_history)
        plt.title("cost value")
        plt.show()
        
        warped_results = list(map(np.squeeze, 
                                  (sess.run(distorImgs))))
        hor1 = np.hstack(warped_results[:2])
        hor2 = np.hstack(warped_results[2:4])
        plt.imshow(np.vstack([hor1, hor2]))
        turn_off_tick_marks()
        plt.show()
        
        distorImgsV, depth_val = sess.run(
            [distorImgs, depth])
        

        
        plt.imshow(distorImgsV[0][0])
        plt.title("a warped scene image")
        plt.show()
        
        plt.title("another warped scene image")
        plt.imshow(distorImgsV[3][0])
        plt.show()
        
        plt.imshow(sess.run(depth[0]))
        plt.title("depth estimate (lighter is closer)")
        plt.show()
        
        plt.hist(depth_val.flatten(), bins=40)
        plt.title("depth histogram")
        plt.show()
        


# In[ ]:


3333
2
`

