# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 00:31:23 2022

@author: Ibrahim
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
import os
import os.path


# path = Path(".")
# path = path.glob("*.jpg")
# images[]

# for imagepath in path.glob("*.jpg"):
#     img=cv2.imread(str)

images=[]
names=[]                    # list contatining  all images
folder = 'Assignment data/data/images/test/'
# filename = '*.jpg'

for filename in os.listdir(folder):

    print(filename)

    img=mpimg.imread(folder+filename)  # reading image (Folder path and image name )

    img=np.array(img)                #

    # img=img.flatten()                # Flatten image 
    
    # img.squeeze().permute(1,2,0)

    images.append(img)
    names.append(filename)       # Appending all images in 'images' list 
    
              

# image = mpimg.imread(r"Assignment data/data/images/test/*.jpg")
# plt.imshow(image)
# plt.show()

for k in range(len(images)):

    image=images[k]
    name=names[k]
    
    plt.imshow(image)
    plt.show()
    
    color1=[np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)]
    color2=[np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)]
    # color3=[np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)]
    
    
    old_error=-1
    new_error=-2
    
    while old_error!=new_error:    
    
        pixels1=[]
        pixels2=[]
        # pixels3=[]
        
        locations1=[]
        locations2=[]
        # locations3=[]
        
        for i in range(len(image)):
            for j in range(len(image[0])):
                difference1=sum(abs(image[i][j]-color1))
                difference2=sum(abs(image[i][j]-color2))
                # difference3=sum(abs(image[i][j]-color3))
            
                if(min(difference1,difference2)==difference1):
                    pixels1.append((image[i][j]).tolist())
                    locations1.append([i,j])
                elif(min(difference1,difference2)==difference2):
                    pixels2.append((image[i][j]).tolist())
                    locations2.append([i,j])
                # else:
                #     pixels3.append((image[i][j]).tolist())
                #     locations3.append([i,j])
                # print(i,j)
                
            
        pixels1=np.array(pixels1)
        pixels2=np.array(pixels2)
        # pixels3=np.array(pixels3)
                
        color1=[sum(pixels1[:,0])/len(pixels1[:,0]),sum(pixels1[:,1])/len(pixels1[:,1]),sum(pixels1[:,2])/len(pixels1[:,2])]
        color2=[sum(pixels2[:,0])/len(pixels2[:,0]),sum(pixels2[:,1])/len(pixels2[:,1]),sum(pixels2[:,2])/len(pixels2[:,2])]
        # color3=[sum(pixels3[:,0])/len(pixels3[:,0]),sum(pixels3[:,1])/len(pixels3[:,1]),sum(pixels3[:,2])/len(pixels3[:,2])]
            
        old_error=new_error
        new_error=0
            
            
            
            
        for i in range(len(pixels1)):
            new_error+=sum(abs(color1-pixels1[i]))
        
        for i in range(len(pixels2)):
            new_error+=sum(abs(color1-pixels2[i]))
        
        # for i in range(len(pixels3)):
        #     new_error+=sum(abs(color1-pixels3[i]))
            
            
            
    
    new_image=np.zeros([len(image),len(image[0]),3])
    
    for i in range(len(locations1)):
        new_image[locations1[i][0],locations1[i][1]]=color1
    
    for i in range(len(locations2)):
        new_image[locations2[i][0],locations2[i][1]]=color2
    
    # for i in range(len(locations3)):
    #     new_image[locations3[i][0],locations3[i][1]]=color3
        
    new_image=np.round(new_image).astype(int)
    plt.imshow(new_image)       

    plt.savefig(r"Assignment data/data/images/results/"+names[k])