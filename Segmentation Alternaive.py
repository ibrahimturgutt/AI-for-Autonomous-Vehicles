# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 23:28:02 2022

@author: İbrahim
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import cv2
import os
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

images=[]                             # list contatining  all images
names=[]
folder = 'Assignment data/data/images/test/'
filename = '*.jpg'

for filename in os.listdir(folder):
    # print(filename)
    image=mpimg.imread(folder+filename)
    image=np.array(image)
    images.append(image)
    names.append(filename)
    
    for z in range(len(images)):
        img=images[z]
        name=names[z]

# img = mpimg.imread('Assignment data/data/images/test/14085.jpg')
# plt.imshow(img)
# imgplot = plt.imshow(img)
# plt.axis('off')
# plt.show()

    X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))
    # print(X)
    
    K = 2
    kmeans = KMeans(n_clusters=K).fit(X)
    label = kmeans.predict(X)
    img4 = np.zeros_like(X)
    
    # replace each pixel by its center
    for k in range(K):
        img4[label == k] = kmeans.cluster_centers_[k]
    
    #reshape and display output image
    img5 = img4.reshape((img.shape[0], img.shape[1], img.shape[2]))
    plt.imshow(img5, interpolation="nearest" )
    plt.axis("off")
    plt.show()
    
    plt.savefig(r"Assignment data/data/images/results/"+names[z])
    
    
    
# X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
# plt.scatter(X[:,0], X[:,1])

# wcss = []
# for i in range(1,11):
#     kemans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

# kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
# pred_y = kmeans.fit_predict(X)
# plt.scatter(X[:,0], X[:,1])
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, c='red')
# plt.show()
