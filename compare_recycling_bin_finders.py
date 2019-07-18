#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 - jnikhil (Nikhil Jamdade)
The progam is tested and working correctly if incase of any issues. please contact jnikhil@seas.upenn.edu
"""
'''Pakages and dependancies to be install before the program runs are scikit-image, numpy, matplotlib '''

import skimage
import os 
import glob
import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import math  
import sys


def intersection_over_union(prediction, ground_truth):
    """
    Compute the IOU between a predicted segmentation and a ground-truth
    :param prediction: A (height, width) boolean array indicating the predicted segmentation
    :param ground_truth: A (height, width) boolean array indicating the correct segmentation
    :return float: The IOU score
    """
    return np.sum(prediction & ground_truth, dtype=float) / np.sum(prediction | ground_truth)


def blue_region_segmenter(rgb_image, relative_blueness_threshold = 1.):
    """
    Segment out the blue regions of the image.
    :param rgb_image: A (height, width, 3) RGB image numpy array
    :param relative_blueness_threshold: The threshold for how strong the blue should be relative to the red+green
    :return: A (height, width) boolean numpy array indicating the pixels occupied by the bin
    """
    labels = rgb_image[:, :, 2] > relative_blueness_threshold * (rgb_image[:, :, 0].astype(float) + rgb_image[:, :, 1])
    return labels


def bluest_rectangle_segmenter(rgb_image, rectangle_size=(120, 80)):
    """
    Segment out the bluest rectangle of the given size in the image.
    :param rgb_image: A (height, width, 3) RGB image numpy array
    :param rectangle_size: The (rect_height, rect_width) of the rectangle which you expect the bin to occupy
    :return: A (height, width) boolean numpy array indicating the pixels occupied by the bin
    """
    ry, rx = rectangle_size
    relative_blueness = rgb_image[:, :, 2].astype(float)/(rgb_image.astype(float).sum(axis=2)+1e-6)
    integral_image = relative_blueness.cumsum(axis=0).cumsum(axis=1)
    blueness_in_box = integral_image[ry:, rx:] - integral_image[:-ry, rx:] - integral_image[ry:, :-rx] + integral_image[:-ry, :-rx]
    row, col = np.unravel_index(blueness_in_box.argmax(), blueness_in_box.shape)
    labels = np.zeros(relative_blueness.shape, dtype=np.bool)
    labels[row:row+ry, col:col+rx] = True
    return labels


def load_dataset():
    
    '''Load the image files and labels fiels and store them in global variables'''
    
    imagefiles = glob.glob(path +"/*.jpg")
    imagefiles.sort()
    images = [io.imread(file) for file in imagefiles]
    labelfiles = glob.glob(path +"/*.png")
    labelfiles.sort()
    labels = [io.imread(file) for file in labelfiles]
    return (images,labels)

def optimum_threshold_parameter():
    
    ''' 
    Relative Blueness Threshold is one of the parameters that can be tuned to optimum to provide optimal perforamnce results for blue_region_segmenter.
    Here, for each threshold value from 0 to 2  with step size of 0.1, all images are segmented using blue_region_segmenter and mean IOU
    is calulated for each threshold value over set of segmented images. 
    best_threshold is the value in the list of thresholds for which average IOU is maximum. 
    '''
    
    Threshold = [("{:1.1f}".format(th))for th in np.arange(0.0,2.0,0.1)]
    Threshold = np.float64(Threshold)
    regseg_avg_iou = []             # store list of average IOUs for region segmenter
    for TH in Threshold: 
        reg_segimg = []               # store list of segmented images for each threshold
        iou_regsegmenter = []         # store list of IOU per image per threshold
        for img in images:
            reg_segimg.append(blue_region_segmenter(img,TH))
        iou_regsegmenter = [intersection_over_union(combination[0],combination[1]) for combination in zip(reg_segimg,labels)]    
        regseg_avg_iou.append(np.mean(iou_regsegmenter))   
   
    th_ind = np.argmax(regseg_avg_iou)      # index of maximum average IOU and threshold
    best_threshold = Threshold[th_ind]     
    return best_threshold


def optimum_rect_size_parameter():
    
    '''
    Size ofrectangle is one the parameter which can be optimised to give optimal performance of bluest_rectangle_segmenter
    Here keeping the aspect ratio equal, area of rectangle and hence height amd width are varied respectively 
    e.g 50% change in area i.e d=1/2, keeping aspect ratio same, both height and width are changed by sqrt(d).   Area = height x width 
    Accordingly value of d is calculated by taking sqrt of values from 0.1 to 2 with step 0.1. 
    Ry and Rx are varied based on d values to segment images using bluest_rectangle_segmenter per rect size per set of images. 
    Mean IOU is calulated for each rect_size over set of segmented images. The best_rect_size is the value which gives maximum average IOU 
    '''
    
    dist_perc = [("{:1.3f}".format(math.sqrt(dpercent)))for dpercent in np.arange(0.1,2.0,0.1)]   # store percentage change in Area
    dist_perc = np.float64(dist_perc) 
    rectseg_avg_iou = []                  # store list of average IOUs for rectangle segmenter
    for d in dist_perc:
        rect_segimg = []                   # store list of segmented images for region segmenter
        iou_rectsegmenter = []             # store list of IOU per image per rect_size
        for img in images:
            ry = int(120*d)
            rx = int(80*d)
            rect_size = (ry,rx)
            rect_segimg.append(bluest_rectangle_segmenter(img,rect_size))
        iou_rectsegmenter = [intersection_over_union(combination[0],combination[1]) for combination in zip(rect_segimg,labels)]
        rectseg_avg_iou.append(np.mean(iou_rectsegmenter))
    dist_ind = np.argmax(rectseg_avg_iou)
    best_dist = dist_perc[dist_ind]
    ry = int(120*best_dist)
    rx = int(80*best_dist)
    best_rect_size = (ry,rx)
    return best_rect_size


def optimum_iou(best_threshold,best_rect_size):
    
    '''
    This function segments images with respective optimum paramer and calculates optimal IOU for each segmenters for each image input
    '''
    test_reg_segimg = [blue_region_segmenter(img,best_threshold) for img in images]
    test_rect_segimg = [bluest_rectangle_segmenter(img,best_rect_size) for img in images]
    test_iou_regseg = [intersection_over_union(combination[0],combination[1]) for combination in zip(test_reg_segimg,labels)]
    test_iou_rectseg = [intersection_over_union(combination[0],combination[1]) for combination in zip(test_rect_segimg,labels)]  
    return (test_iou_regseg,test_iou_rectseg)     #returns optimum IOU list for each images using region segmenter and rectanlge segmenter 


def main():
    
    best_threshold = optimum_threshold_parameter()       # best threshold   
    best_rect_size = optimum_rect_size_parameter()       # best rectagle size    
    test_iou_regseg,test_iou_rectseg=optimum_iou(best_threshold,best_rect_size)        #optimum IOUs for each image using each segmenter
    
    
    # printing text-display indicating the performance of under Blue Region Segmenter and its best parameter, both per-image and on average
    print("\nThe optimum threshold parameter for Blue Region Segmenter: " + str(best_threshold)) 
    print("\nThe performace of Blue Region Segmenter on each image out of "+ str(len(images))+ " images \nwith respect to optimum threshold in terms of IOU:\n") 
    for i in range (len(images)):
        print ("Image "+str(i+1)+": " + str("{:1.3f}".format(test_iou_regseg[i]))) 
    print("The average IOU on images for Blue Region Segmenter:" + str("{:1.3f}".format(np.mean(test_iou_regseg))))
    
    
    # printing text-display indicating the performance of under Bluest Rectangle Segmenter and its best parameter, both per-image and on average
    print("\n\nThe optimum rectangle size parameter for Bluest Rectangle Segmenter: " + str(best_rect_size)) 
    print("\nThe performace of Bluest rectangle Segmenter on each image out of "+ str(len(images))+ " images \nwith respect to optimum rectangle size in terms of IOU:\n") 
    for i in range (len(images)):
        print ("Image "+str(i+1)+": " + str("{:1.3f}".format(test_iou_rectseg[i])))  
    print("The average IOU on images for Bluest Rectangle Segmenter:" + str("{:1.3f}".format(np.mean(test_iou_rectseg))))  
   
  
    # Command Line Input from users
    char = (input('\n\nDo you want select images to compare performance of segmenters (Y/N): '))
    
    while((char != 'N')):    
        if (char == 'Y'):    
            valid = False 
            while(not valid):
     
                user_input = input("Select Images from 1 to "+str(len(images)) + " to Compare e.g 1,2,6,9,11 : ")  
                input_list = user_input.split(',')
               
                if (len(input_list) != len([x for x in input_list if x.isdigit()])):      # checks if user input are numbers or not 
                    print('Enter comma seperated intigers only')
                    continue
                
                image_numbers = [int(x.strip()) for x in input_list]      #  collects comma seperated user input into list and convert element into int          
                valid = True
                
                if (0 in image_numbers):
                    print("0 cant be a input, select images from 1 to "+ str(len(images)))
                    valid = False
                
                
                elif (max(image_numbers)>len(images)):
                    print("Database contains "+ str(len(images)) +" images. "+ "You have entered image number greater than "+str(len(images)))
                    valid = False
                
                elif (len(image_numbers) != len(set(image_numbers))):
                    print("You have entered duplicate images/numbers. Do not enter same image/number more than once")
                    valid=False
            
            #Calculating IOU for images selected by user with respect to optimal parameters for each segmenters
            images_disp = [images[(number-1)] for number in image_numbers]    # images selected by user
            labels_disp = [labels[(number-1)] for number in image_numbers]    # labels for selected images 
           
            reg_seg_disp = [blue_region_segmenter(img,best_threshold) for img in images_disp] 
            rect_seg_disp = [bluest_rectangle_segmenter(img,best_rect_size) for img in images_disp]
            opt_iou_reg_seg = [intersection_over_union(combination[0],combination[1]) for combination in zip(reg_seg_disp,labels_disp)] 
            opt_iou_rect_seg = [intersection_over_union(combination[0],combination[1]) for combination in zip(rect_seg_disp,labels_disp)] 
            
           
            print("Close figure window to proceed")
           
            # plotting and visually comparing the performance of each of the optimal-segmenters 
            plt.rc('font', size=10)
            plt.rc('axes', titlesize=10)
            plt.figure(figsize=(10,8))
            xaxis_list = ["image "+str(number) for number in image_numbers]
            plt.scatter(xaxis_list,opt_iou_reg_seg, c='b', s=100,alpha=0.7,label='Blue Region Segmenter')
            plt.scatter(xaxis_list,opt_iou_rect_seg, marker='^', c='r',s=100, alpha=0.5,label='Bluest Rectangle Segmenter')
            plt.title('Performance comparision of Blue Region Segmenter \nand Bluest rectangle Segmenter using IOU', y=1.1)
            plt.ylabel('IOU')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
            plt.show()
            
            char = input('Do you want to compare performance of segmenters on more images (Y/N): ')
            
        else:   
            char = (input('Type Y or N:'))
        
                
if __name__ == "__main__":
    path = (sys.argv[1])         # command line arguments: path for images and labels folder
    images,labels = load_dataset()      # global varibles images and labels 
    labels = np.array(labels, dtype=bool)      #coverting labels to numpy boolean 
    main()
    