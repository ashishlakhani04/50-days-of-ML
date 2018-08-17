import numpy as np
import cv2
import os
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

""" Prepare Data """
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig/255.
X_test = X_test_orig/255.

Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

""" Create CNN Model"""
def HappyModel(input_shape):
    
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    X_input = Input(input_shape)
    
    X = ZeroPadding2D((3,3))(X_input)
    
    X = Conv2D(32,(7,7),strides=(1,1),name='conv0')(X)
    X = BatchNormalization(axis=3,name='bn0')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2, 2), name='max_pool')(X)
    X = Flatten()(X)
    X = Dense(1, activation='relu', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model

""" Train the model """
happyModel = HappyModel(X_train.shape[1:])

happyModel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

happyModel.fit(X_train, Y_train, epochs=5, batch_size=50)



cap = cv2.VideoCapture(0)

# Load the haar cascade for frontal face
face_cascade = cv2.CascadeClassifier('datasets/haarcascade_frontalface_alt.xml')

while True:

	ret, frame = cap.read()
	if ret == False:
		continue

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	face_section=None
	for face in faces:
		x, y, w, h = face

		# Get the face ROI
		offset = 7
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset,:]
		face_section = cv2.resize(face_section, (64, 64))


		# print(face_section.shape)
		face_section = np.expand_dims(face_section, axis=0)
		face_section = preprocess_input(face_section)
		print(happyModel.predict(face_section))
		# out = knn(trainset, face_section.flatten())

		# Draw rectangle in the original image
		# cv2.putText(img = frame, text = names[int(out)], org=(x,y-10), fontFace =  font, fontScale = 1, color = (255,0,0),thickness = 2,lineType= cv2.LINE_AA)
		# cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)
		# cv2.putText(frame, names[int(out)],(x,y-10), font, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

	
	cv2.imshow("Faces", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()



