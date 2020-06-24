from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern, hog
import cv2

# Keras
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from keras.models import load_model
# from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

def normalize(X):
    norm = StandardScaler()
    X_norm = norm.fit_transform(X)
    return X_norm

def lbp(X, P, R, method = 'uniform'):
    m = X.shape[0]
    if(method == 'uniform'):
        p = P + 1
    elif(method == 'default'):
        p = np.power(2,P) - 1
    elif method =='nri_uniform':
        p = 134       ##12:134, 16:242
    hist_X = np.zeros((m, p+1))
    for i in range(m):
        im = X[i]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        eps = 1e-8
        lbp = local_binary_pattern(gray, P = P, R = R, method=method)
        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, p+2),range=(0, p+1))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        hist_X[i] = hist
    return hist_X

def hog_feature(X):
    X_hog = []
    m = X.shape[0]
    for i in range(m):
        fd= hog(X[i], orientations=12, pixels_per_cell=(64, 64), 
                 cells_per_block=(2,2), visualize=False, multichannel=True, block_norm='L2')
        X_hog.append(fd)
    return np.asarray(X_hog)  
    

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model/clf2.sav'

# Load your trained model
#model = load_model(MODEL_PATH)
 #model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
class_name = ['bicycle', 'bus', 'car', 'motorbike']
model = pickle.load(open(MODEL_PATH, 'rb'))
print('Model loaded. Check http://127.0.0.1:5000/ or http://localhost:5000/')


def model_predict(img_path, model):
    #img = image.load_img(img_path, target_size=(224, 224))
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256,256))
    img = np.expand_dims(img, axis = 0)
    X_lbp = lbp(img, P=12, R =2, method = 'nri_uniform')
    X_hog = hog_feature(img)
    X_lbp_norm = normalize(X_lbp.T).T
    X_hog_norm = normalize(X_hog.T).T
    X_total = np.concatenate((X_lbp_norm, X_hog_norm), axis = 1)
    print(X_total.shape)
    # Preprocessing the image
    #x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.predict(X_total)
    print('pred.shape: ', preds.shape)
    return preds[0]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        result = class_name[preds]  # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
    app.run()
