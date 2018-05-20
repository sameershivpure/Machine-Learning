import numpy as np
import cv2 as cv
import os
import glob
from sklearn import svm
from sklearn.externals import joblib


pos_im_path = "./Images/Train/cars"
neg_im_path = "./Images/Train/noncars"
hog = cv.HOGDescriptor('Shivpure_hogxml_01.xml')
features = []
labels = []

for im_file in glob.glob(os.path.join(pos_im_path, "*")):
    img = cv.imread(im_file)
    img = cv.GaussianBlur(img, (3, 3), 0)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    histogram = hog.compute(img, winStride=(8, 8),padding=(2,2))
    features.append(histogram)
    labels.append(1)

for im_file in glob.glob(os.path.join(neg_im_path, "*")):
    im = cv.imread(im_file)
    im = cv.GaussianBlur(im, (3, 3), 0)
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    histogram = hog.compute(im, winStride=(8, 8),padding=(2,2))
    features.append(histogram)
    labels.append(0)

trainData = np.float32(features).reshape(-1,len(features[0]))
responses = np.int32(labels).reshape(-1,1)

SVM = svm.SVC(kernel='sigmoid')
SVM.fit(trainData, labels)
joblib.dump(SVM, 'Shivpure_svm_01.data')