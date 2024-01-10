import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
import glob
import tqdm

def create_class_mask(img, color_map, is_normalized_img=True, is_normalized_map=False, show_masks=False):
    """
    Function to create C matrices from the segmented image, where each of the C matrices is for one class
    with all ones at the pixel positions where that class is present

    img = The segmented image

    color_map = A list with tuples that contains all the RGB values for each color that represents
                some class in that image

    is_normalized_img = Boolean - Whether the image is normalized or not
                        If normalized, then the image is multiplied with 255

    is_normalized_map = Boolean - Represents whether the color map is normalized or not, if so
                        then the color map values are multiplied with 255

    show_masks = Wherether to show the created masks or not
    """

    if is_normalized_img and (not is_normalized_map):
        img *= 255

    if is_normalized_map and (not is_normalized_img):
        img = img / 255
    
    mask = []
    hw_tuple = img.shape[:-1]
    for color in color_map:
        color_img = []
        for idx in range(3):
            color_img.append(np.ones(hw_tuple) * color[idx])

        color_img = np.array(color_img, dtype=np.uint8).transpose(1, 2, 0)

        mask.append(np.uint8((color_img == img).sum(axis = -1) == 3))

    return np.array(mask)


def loader(training_path, segmented_path, batch_size, h=512, w=512):
    """
    The Loader to generate inputs and labels from the Image and Segmented Directory

    Arguments:

    training_path - str - Path to the directory that contains the training images

    segmented_path - str - Path to the directory that contains the segmented images

    batch_size - int - the batch size

    yields inputs and labels of the batch size
    """
    #print(training_path)
    filenames_t = glob.glob(training_path+f'*/*.jpg')
    #print(filenames_t)
    total_files_t = len(filenames_t)

    filenames_s = glob.glob(segmented_path+f'*/*.png') 
    total_files_s = len(filenames_s)#contain only png file names
    
    #print(filenames_s)
    print(total_files_s, total_files_t)
    assert(total_files_t == total_files_s)
    
    if str(batch_size).lower() == 'all':
        batch_size = 500
    
    print(batch_size)
    road_idx = 0
    while(1):
        #print('inside loader')
        batch_idxs = np.random.randint(0, total_files_s, batch_size)
            
        #print(len(batch_idxs))
        inputs = []
        labels = []
        
        for jj in batch_idxs:
            # Reading the training image using OpenCV
            img = cv2.imread(filenames_t[jj])
            img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)
            inputs.append(img)
    
            # Reading the segmented image using OpenCV
            label_img = cv2.imread(filenames_s[jj], cv2.IMREAD_UNCHANGED)
            
            road_mask = (label_img == road_idx).astype(np.uint8)
            road_mask = cv2.resize(road_mask, (h, w), cv2.INTER_NEAREST)
            labels.append(road_mask)
        
        #print(labels)
        #print(len(inputs))
        #print(len(labels))
        inputs = np.stack(inputs, axis=2)
        inputs = torch.tensor(inputs).transpose(0, 2).transpose(1, 3)
        labels = torch.tensor(labels)
        
        yield inputs, labels

def decode_segmap(image):
    road_color = [255, 0, 255]  # Magenta for road
    non_road_color = [255, 0, 0]  # Red for non-road

    # Create an RGB image with all pixels set to the non-road color
    rgb = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * non_road_color

    # Set pixels belonging to the road to the road color
    rgb[image == 1] = road_color

    return rgb

def show_images(images, in_row=True):
    '''
    Helper function to show 3 images
    '''
    total_images = len(images)

    rc_tuple = (1, total_images)
    if not in_row:
        rc_tuple = (total_images, 1)
    
	#figure = plt.figure(figsize=(20, 10))
    for ii in range(len(images)):
        plt.subplot(*rc_tuple, ii+1)
        plt.title(images[ii][0])
        plt.axis('off')
        plt.imshow(images[ii][1])
    plt.show()


def get_class_weights(loader, num_classes, c=1.02):
    '''
    This class return the class weights for each class
    
    Arguments:
    - loader : The generator object which return all the labels at one iteration
               Do Note: That this class expects all the labels to be returned in
               one iteration

    - num_classes : The number of classes

    Return:
    - class_weights : An array equal in length to the number of classes
                      containing the class weights for each class
    '''
    #print('inside ge_class_weights')
    _, labels = next(loader)
    all_labels = labels.flatten()
    each_class = np.bincount(all_labels, minlength=num_classes)
    #print(len(each_class))
    prospensity_score = each_class / len(all_labels)
    class_weights = 1 / (np.log(c + prospensity_score))
    #print('finished')
    return class_weights
