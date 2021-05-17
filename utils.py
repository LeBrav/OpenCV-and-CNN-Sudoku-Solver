import numpy as np
import cv2
from sys import exit

'https://github.com/UB-Mannheim/tesseract/wiki'
'pip install pytesseract'
import pytesseract
import tensorflow as tf
from tensorflow.keras.models import load_model

from matplotlib import pyplot as plt
from skimage import morphology

import joblib
import os
import pickle
import keras
# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Model, Sequential, load_model
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import BatchNormalization, Input
from keras.optimizers import Adam
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd
import math

#used to add pad in an image
def pad_with(vector, pad_width, iaxis, kwargs):  
    padding_value = kwargs.get('padder', 255)  
    vector[:pad_width[0]] = padding_value  
    vector[-pad_width[1]:] = padding_value 

#it returns de second most common value of pixels in an image
def find_second_most_common_pixel(image):
    histogram = {}  

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pixel_val = image[x][y]
            if pixel_val in histogram:
                histogram[pixel_val] += 1 
            else:
                histogram[pixel_val] = 1
                
    key_to_delete = max(histogram, key = histogram.get)
    del histogram[key_to_delete]
    mode_pixel_val = max(histogram, key = histogram.get)
    
    #in case its not the border dont delete anything
    if histogram[mode_pixel_val] < 500:
        mode_pixel_val = 0
    
    return mode_pixel_val

#it returns de second most common value of pixels in an image, that its not the border
def find_second_most_common_pixel_no_borders(image):
    histogram = {}  

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pixel_val = image[x][y]
            if pixel_val in histogram:
                histogram[pixel_val] += 1 
            else:
                histogram[pixel_val] = 1
                
    key_to_delete = max(histogram, key = histogram.get)
    del histogram[key_to_delete]
    if histogram: #not empty
        mode_pixel_val = max(histogram, key = histogram.get)
    else:
        mode_pixel_val = -1
    return mode_pixel_val

#This function finds the borders of the image and delete them
def fix_borders(channel, max_crop_perc, line_border_perc, edge_detector=' '):
    
    "https://gyazo.com/c015412c84868f8f0bba9f8729029989"
    if edge_detector == 'sobel':
        #sobel
        img_sobelx = cv2.Sobel(channel,cv2.CV_8U,1,0,ksize=3)
        img_sobely = cv2.Sobel(channel,cv2.CV_8U,0,1,ksize=3)
        edges = img_sobelx + img_sobely
    else:
        #canny
        edges = cv2.Canny(channel,100,200, 3)
    
    '''
    channel = np.delete(channel, range(1), axis = 0)
    channel = np.delete(channel, range(channel.shape[0]-1, channel.shape[0]), axis = 0)
    channel = np.delete(channel, range(1), axis = 1)
    channel = np.delete(channel, range(channel.shape[1]-1, channel.shape[1]), axis = 1)
    channel = cv2.copyMakeBorder(channel, 1, 1, 1, 1, cv2.BORDER_CONSTANT,value=255)
    
    contours, _ = cv2.findContours(channel,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    
    # Iterate through all the contours
    for contour in contours:
        # Find bounding rectangles
        x,y,w,h = cv2.boundingRect(contour)
        # Draw the rectangle
        cv2.rectangle(channel,(x,y),(x+w,y+h),0,1)
    '''   
    edges = channel>0 #convert to binary

    #edges = edges >0
    
    
    #To find the border we are going to first treat binary speaking the image to emphazise 
    #the border of the iamge. Then we are going to obtain the index (np.where) of the
    #part of the image (limited by max_crop_perc), that have more than "line_border_perc"
    #white pixels, that mean they are borders
    
    suma_columnes = np.sum(edges, axis = 1)
    #Top border
    index_fila_amunt = np.where(suma_columnes[:int(channel.shape[0]*max_crop_perc)] > int(channel.shape[0]*line_border_perc))[0]
    if index_fila_amunt.size != 0:
        index_fila_amunt = index_fila_amunt[-1]
        channel = np.delete(channel, range(index_fila_amunt+1), axis = 0)

    suma_columnes = np.sum(edges, axis = 1)
    #bottom border
    index_fila_avall = np.where(np.flip(suma_columnes[-int(channel.shape[0]*max_crop_perc):]) > int(channel.shape[0]*line_border_perc))[0]
    if index_fila_avall.size != 0:
        index_fila_avall = index_fila_avall[-1]
        channel = np.delete(channel, range(channel.shape[0]-index_fila_avall-1, channel.shape[0]), axis = 0)

    suma_files = np.sum(edges, axis = 0)    
    #left border
    index_col_esquerra = np.where(suma_files[:int(channel.shape[1]*max_crop_perc)] > int(channel.shape[1]*line_border_perc))[0]
    if index_col_esquerra.size != 0:
        index_col_esquerra = index_col_esquerra[-1]
        channel = np.delete(channel, range(index_col_esquerra+1), axis = 1)
        
    suma_files = np.sum(edges, axis = 0)
    #right border
    index_col_dreta = np.where(np.flip(suma_files[-int(channel.shape[1]*max_crop_perc):]) > int(channel.shape[1]*line_border_perc))[0]
    if index_col_dreta.size != 0:
        index_col_dreta = index_col_dreta[-1]
        channel = np.delete(channel, range(channel.shape[1]-index_col_dreta-1, channel.shape[1]), axis = 1)
    
       
    return channel

#this function adds padding to the image until resolution 495
def resize_img_padding(img):
    
    delta_w = 495 - img.shape[1]
    delta_h = 495 - img.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    #if top bottomm left or right is negative, make it 0
    top = max(0,top)
    bottom = max(0,bottom)
    left = max(0,left)
    right = max(0,right)
    #add padding to the image to make the img 32x32
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
    img = img.astype("uint8")
    return img

#binarize input image applying gaussian blue and adaptative thresholding
#and ereasing the border suing conected pixels and common pixels
def get_binary_img(img):
    #img = cv2.blur(img,(2,2))
    img = cv2.GaussianBlur(img, (5, 5), 1)  # ADD GAUSSIAN BLUR
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,33,22)
    
    img[img<255] = 0
    
    img = cv2.bitwise_not(img)#change black to white
    img = fix_borders(img, 1/30, 0.2)#crop borders
    img = resize_img_padding(img)#resize de img to 495x495
    
    
    retval, img = cv2.connectedComponents(img)
    img = img.astype("uint8")
    '''
    plt.imshow(img)
    plt.show()
    '''
    #elements in img have lower value if the connected component is bigger
    #if we get the second most common pixel value, this will be the borders (as the most common one will be background)
    #and if we binarize the img not taking into account the borders, this will be easier to preprocess later  
    borders_value = find_second_most_common_pixel(img)
    img[img>borders_value] = 255
    img[img<=borders_value] = 0
    
    #remove isolted group of pixels less than min size
    retval, labels = cv2.connectedComponents(img)
    img = morphology.remove_small_objects(labels, min_size=50, connectivity=2)
    img = img.astype("uint8")
        
    img[img>0] = 255
    
    img = cv2.bitwise_not(img)
    return img

#divide the processed img into boxes (sudoku's cells)
def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes=[]
    for row in rows:
        cols= np.hsplit(row,9)
        for box in cols:
            boxes.append(box)
    return boxes

#delete all the image not included in x,y and width w, height h
#used to keep only the number of the sudoku
def crop_border(box, x,y, w, h):
    box = np.delete(box, range(y), axis = 0)
    box = np.delete(box, range(h, box.shape[0]), axis = 0)
    box = np.delete(box, range(x), axis = 1)
    box = np.delete(box, range(w, box.shape[1]), axis = 1)
    
    return box

#this function does the preprocessing of every box (sudoku cell)
def image_proc_boxes(boxes):
    for i, box in enumerate(boxes):
        
        
        box = cv2.bitwise_not(box)#change black to white
        
        #remove isolted group of pixels less than min size
        retval, labels = cv2.connectedComponents(box)
        box = morphology.remove_small_objects(labels.astype('bool'), min_size=50, connectivity=2)
        box = box.astype("uint8")
        box[box>0] = 255
        
        #crop borders
        box = fix_borders(box, 1/5, 0.5)
        
        
        #remove isolted group of pixels less than min size, the leftovers of borders
        retval, labels = cv2.connectedComponents(box)
        box = morphology.remove_small_objects(labels.astype('bool'), min_size=50, connectivity=2)
        box = box.astype("uint8")
        box[box>0] = 255

        #once we only have the digit left, center it
        x,y,w,h = cv2.boundingRect(box) #get bounding box of number
        box = crop_border(box,x,y,w,h) #get only the number in the box
        
        #resize de box to 32
        delta_w = 32 - box.shape[1]
        delta_h = 32 - box.shape[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        
        #if top bottomm left or right is negative, make it 0
        top = max(0,top)
        bottom = max(0,bottom)
        left = max(0,left)
        right = max(0,right)
        #add padding to the image to make the img 32x32
        box = cv2.copyMakeBorder(box, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
        box = box.astype("uint8")
        
        #number is occuping all box, resize it
        if delta_w+delta_h < 20: 
            box = cv2.resize(box,(26,26))
            delta_w = 32 - box.shape[1]
            delta_h = 32 - box.shape[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            box = cv2.copyMakeBorder(box, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
            box = box.astype("uint8")
            
        '''
        if box.shape[0] > 32 or box.shape[1] > 32:
            delta_w = 32 - box.shape[1]
            delta_h = 32 - box.shape[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            #if top bottomm left or right is negative, make it 0
            top = max(0,top)
            bottom = max(0,bottom)
            left = max(0,left)
            right = max(0,right)
            box = cv2.copyMakeBorder(box, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
            box = box.astype("uint8")
            kernel = np.ones((3,3),np.uint8)
            box = cv2.dilate(box,kernel,iterations = 1)
            box = cv2.resize(box,(28,28))
        '''

        
          
        
        
        box[box>0] = 255
        
        
        retval, box = cv2.connectedComponents(box)
        box = box.astype("uint8")
        box = cv2.GaussianBlur(box, (5, 5), 1)  # ADD GAUSSIAN BLUR
        #remove isolted group of pixels less than min size
        retval, labels = cv2.connectedComponents(box)
        box = morphology.remove_small_objects(labels.astype('bool'), min_size=50, connectivity=2)
        box = box.astype("uint8")
        box = cv2.bitwise_not(box)#change black to white
        
        '''
        plt.imshow(box)
        plt.show()
        '''
        #make the image equal to the input of the CNN
        box[box<255] = 0
        box = box / 255
        
        
        boxes[i] = box
    boxes_np = np.asarray(boxes)
     
    boxes_np2 = boxes_np.reshape(boxes_np.shape[0], 32, 32, 1)
    
    boxes_np = boxes_np.astype('uint8')
    return boxes_np2, boxes_np

#this function plots the preprocessed boxes in one single plot
def plot_boxes(boxes_np, name):
    w=32
    h=32
    
    fig=plt.figure(figsize=(8, 8))
    plt.title("Processed Boxes")
    plt.axis('off')
    columns = 9
    rows = 9
    for i in range(1, columns*rows + 1):
        img = boxes_np[i-1]
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(img)
    plt.savefig('results/' +name + '/'+'3_plot_boxes.png', dpi=300, bbox_inches='tight')
    plt.show()

#this function plots the result sudoku prediction of the models, and its confidence
def plot_prob(boxes_prob, numbers, name):
    
    w=32
    h=32
    fig=plt.figure(figsize=(8, 8))
    plt.title("Model Prediction Result")
    plt.axis('off')
    columns = 9
    rows = 9
    for i in range(1, columns*rows + 1):
        text = str(boxes_prob[i-1])
        text = text[0:5]
        fig.add_subplot(rows, columns, i)
        plt.text(0.5, 0.5, str(numbers[i-1]), fontsize=20)
        if boxes_prob[i-1] >0.99:
            plt.text(0.1, 0.1, text, bbox=dict(facecolor='green', alpha=0.5), fontsize=12)
        else:
            if boxes_prob[i-1] >0.75:
                plt.text(0.1, 0.1, text, bbox=dict(facecolor='yellow', alpha=0.5), fontsize=12)
            else:
                if boxes_prob[i-1] >0:
                    plt.text(0.1, 0.1, text, bbox=dict(facecolor='red', alpha=0.5), fontsize=12)
                else:
                    plt.text(0.1, 0.1, text, bbox=dict(facecolor='gray', alpha=0.5), fontsize=12)
        
        #plt.axis('off')
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
    
    plt.savefig('results/' +name + '/'+ '4_plot_prob.png', dpi=300, bbox_inches='tight')
    plt.show()


#using both models and tesseract, predict the output (number) and its confidence
def Predict(model1, model2, boxes_np, boxes, custom_config):
    prediction=model1.predict(boxes_np)
    prediction2=model2.predict(boxes_np)
    
    numbers = []
    boxes_prob = []
    for i,box in enumerate(boxes):
        predict_prob = max(prediction[i])
        predict_prob2 = max(prediction2[i])
        #######tesseract#######
        string = pytesseract.image_to_data(box, config=custom_config, output_type='data.frame')
        aux =string['conf'].to_numpy()
        argmax = np.argmax(aux)
        predict_prob3 = string['conf'][argmax]
        predict_prob3 = predict_prob3/100
        ##########################
        predict_prob = max(predict_prob, predict_prob2, predict_prob3)
        #print(predict_prob, predict_prob2, predict_prob3)
        #print(np.argmax(prediction[i]),np.argmax(prediction2[i]), string['text'][argmax])
        if predict_prob < 0.50:
            numbers.append(0)
            boxes_prob.append(-1)
        else:
            if predict_prob == predict_prob: 
                    numbers.append(np.argmax(prediction[i]))
            else:
                if predict_prob == predict_prob2:
                    numbers.append(np.argmax(prediction2[i]))
                else:
                    numbers.append(int(string['text'][argmax]))
                    
            boxes_prob.append(predict_prob)
    
    
    return boxes_prob, numbers
###########################ARTIFICIAL INTELLIGENCE###########################
#Check if the number we want to put in a cell is a possible move
def AvaliablePlace(num, nivell, sudoku):
    #check if num is in row
    m_Dim = int(sudoku.shape[0]/9)
    m_n = int(m_Dim/3)
    for i in range(m_Dim):
        if sudoku[int(nivell - int((nivell % m_Dim)) + i)] == num:
            return False

	#check if num is in col
    for i in range(m_Dim):
        if sudoku[int(int((nivell % m_Dim)) + i * m_Dim)] == num:
            return False

	#check if num is in 3x3 block
    auxf = int(int(int(nivell / m_Dim) / m_n) * m_n)
    auxc = int(int(int(nivell % m_Dim) / m_n) * m_n)
    for i in range(m_n):
        for j in range(m_n):
            if sudoku[int(j + i * m_Dim + auxf * m_Dim + auxc)] == num:
                return False
                
    return True

#Backtracking algorithms that solve the Sudoku
def Solve(nivell, sudoku):
    m_Dim = int(sudoku.shape[0]/9)
    if nivell < int(sudoku.shape[0]):
        if sudoku[nivell] == 0:
            for i in range(1,m_Dim+1):
                if AvaliablePlace(i, nivell, sudoku):
                    sudoku[nivell] = i
                    solucio = Solve(nivell + 1, sudoku)
                    if solucio:
                        return True
                    else:
                        sudoku[nivell] = 0
            return False
        else:
            solucio = Solve(nivell + 1, sudoku)
            return solucio
    return True
#############################################################################

#load digit imgs
def Load_digits():
    digits = []
  
    images_of_number = os.listdir('digits')
    for y in images_of_number:
        img_data = cv2.imread('digits'+"/"+y)
        img_data = cv2.resize(img_data,(32,32))
        digits.append(img_data[:,:,1])
    return digits

#overlays the solution of the numbers
def overlay_solution(img_c, sudoku_solution):
    zeros = np.zeros((32,32))
    #Load digit numpy images 32x32
    dig = Load_digits()
    for x in range(0,495,55):
        for y in range(0,495,55):
            num_sol = sudoku_solution[int(x/55)][int(y/55)]
            if num_sol != 0:
                img_c[x+10:x+32+10,y+10:y+32+10,1] = dig[num_sol-1]
                img_c[x+10:x+32+10,y+10:y+32+10,0] = zeros
                img_c[x+10:x+32+10,y+10:y+32+10,2] = zeros
                
    return img_c