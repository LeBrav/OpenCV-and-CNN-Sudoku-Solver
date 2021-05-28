from utils import *

######################PARAMETERS########################
path = 'models/myData'
testPer = 0.2
valPer = 0.2
imageDimensions= (32,32,3)
batchSizeVal= 50
epochsVal = 10
stepsPerEpochVal = 2000
##########################################################
 
####LOAD DATASET##########################################
count = 0
images = []     # list with all the images of the dataset
classNo = []    # list with all the labels of the images

myList = os.listdir(path)
noOfClasses = len(myList)
print("Number of Classes:",noOfClasses)

print("Importing Classes .......")
for x in range (1,noOfClasses):
    images_of_number = os.listdir(path+"/"+str(x))
    for y in images_of_number:
        img_data = cv2.imread(path+"/"+str(x)+"/"+y)
        img_data = cv2.resize(img_data,(32,32))
        images.append(img_data)
        classNo.append(x)
    print(x,end= " ")
print(" ")
print("Total Images = ",len(images))
print("Total labels of image = ",len(classNo))

images = np.array(images)
classNo = np.array(classNo)
print("Total imanges numpy dataset shape = ",images.shape)
##########################################################

####SPLIT DATASET#########################################
X_train,X_test,y_train,y_test = train_test_split(images,classNo,test_size=testPer)
X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=valPer)
print("X_train shape = ",X_train.shape)
print("X_test shape = ",X_test.shape)
print("X_validation shape = ",X_validation.shape)
##########################################################
 
#### PREPOSSES IMAGES FOR TRAINING########################
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img[img<255] = 0
    img = img/255
    return img
 
X_train= np.array(list(map(preProcessing,X_train)))
X_test= np.array(list(map(preProcessing,X_test)))
X_validation= np.array(list(map(preProcessing,X_validation)))
########################################################## 
 
####RESHAPE DATASETS FOR THE INPUT OF THE CNN#############
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
##########################################################
 
####IMAGE AUGMENTATION (get more information from images)#
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
###########################################################

#### ONE HOT ENCODING OF MATRICES##########################
y_train = to_categorical(y_train,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)
##########################################################
 
####CREATING THE CNN STRUCTURE OF OUR MODEL###############
def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2,2)
    noOfNodes= 500
 
    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(imageDimensions[0],
                      imageDimensions[1],1),activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    mod. Perel.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))
 
    model.add(Flatten())
    model.add(Dense(noOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
 
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model
 
model = myModel()
print(model.summary())
##########################################################
 
####TRAIN THE MODEL#######################################
history = model.fit_generator(dataGen.flow(X_train,y_train,
                                 batch_size=batchSizeVal),
                                 steps_per_epoch=int(X_train.shape[0]/batchSizeVal),
                                 epochs=epochsVal,
                                 validation_steps = int(X_validation.shape[0]/batchSizeVal),
                                 validation_data=(X_validation,y_validation),
                                 shuffle=1)
##########################################################
 
#### PLOT THE RESULTS#####################################
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
##########################################################
 
#### EVALUATE USING TEST IMAGES
score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])
 
#### SAVE THE TRAINED MODEL###############################
#model.save('models/digit_deta.h5')
#model.save('models/mnist_model_toxo.h5')
##########################################################