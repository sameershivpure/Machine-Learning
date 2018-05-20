import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from numpy.linalg import svd

# task 1
def display_images(img, row, col, ftitle=""):

    if row == 1 and col == 1:
        plt.imshow(cv.resize(img, (100,100)))
        plt.title(ftitle)

    elif col == 2 :
        figs, axes = plt.subplots(row,col)
        figs.set_size_inches(50, 10)
        figs.suptitle(ftitle)
        for i in range(row):
            for j in range(col):
                axes[i, j].imshow(cv.resize(img[(i * 2 + j)], (100, 100)), cmap=plt.cm.gray)
    else:
        figs, axes = plt.subplots(row, col)
        figs.set_size_inches(10, 5)
        figs.suptitle(ftitle)
        for i in range(row):
            for j in range(col):
                axes[i,j].imshow(cv.resize(img[(i*row + j)], (100,100)), cmap=plt.cm.gray)

    plt.show()

train_images = "./Eigenfaces/Train/"
test_images = "./Eigenfaces/Test/"
images = []
train_img_names = []
for img_file in os.listdir(train_images):
    file_name = os.path.join(train_images, img_file)
    train_img_names.append(img_file.split("_")[0])
    img = (cv.imread(file_name, 0)) / 255.
    images.append(img)

images = np.array(images)
img_vec = images.reshape(images.shape[0],-1)
mean_face = np.mean(images, axis=0)
zerom_img = img_vec - mean_face.reshape(-1)
sd = np.std(zerom_img)
zerom_img /= sd
#display_images(images, 5, 5)
#display_images(mean_face, 1,1)

cov_mat = np.cov(zerom_img)
egval, egvec = np.linalg.eig(cov_mat)
egfaces = np.transpose(np.dot(np.transpose(zerom_img), egvec))
nm= np.linalg.norm(egfaces, axis=1)
nm= nm.reshape(-1,1)
egfaces /= nm
egfaces,s,v = svd(np.transpose(zerom_img), full_matrices=False)
eg_index = np.argsort(egval)[::-1]
#display_images(egfaces.reshape(images.shape),5,5)

# task 2
components = [2, 5, 15]
eg_faceset = []
prj_face = []
for comp in components:
    top_egfaces = []
    for k in range(comp):
        top_egfaces.append(egfaces[eg_index[k]])

    eg_faceset.append(top_egfaces)
    top_egfaces = np.array(top_egfaces)
    projected_faces = np.dot(zerom_img, np.transpose(top_egfaces))
    prj_face.append(projected_faces)
    rec_images = mean_face.reshape(-1) + np.dot(projected_faces, top_egfaces)
    rec_images /= np.std(rec_images)
    #display_images(rec_images.reshape(images.shape),5,5)

# task 3
def get_egfaces(components = [2, 5, 15]):
    eg_faceset = []
    prj_face = []
    for comp in components:
        top_egfaces = []
        for k in range(comp):
            top_egfaces.append(egfaces[eg_index[k]])

        eg_faceset.append(top_egfaces)
        top_egfaces = np.array(top_egfaces)
        projected_faces = np.dot(zerom_img, np.transpose(top_egfaces))
        prj_face.append(projected_faces)

    return eg_faceset, prj_face

timages = []
tlabels = []
blnkimg = np.ones((images.shape[1],images.shape[2]))
#blnkimg /= 2.
for img_file in os.listdir(test_images):
    file_name = os.path.join(test_images, img_file)
    img = (cv.imread(file_name, 0)) / 255.
    img = cv.resize(img,(images.shape[1],images.shape[2]))
    timages.append(img)
    if img_file.split("_")[0] in train_img_names:
        tlabels.append(1)
    else:
        tlabels.append(-1)

tlabels = np.array(tlabels)
tlabels[1:]
tlabels[-7:] = -1
timages = np.array(timages)
zm_timg = timages - mean_face
zm_timg /= sd
timg_vec = zm_timg.reshape(timages.shape[0],-1)

c= [25]
eg_faceset, prj_face = get_egfaces(c)
dist = []
for fin, egf in enumerate(eg_faceset):
    timg_proj = np.dot(timg_vec, np.transpose(egf))
    pred = []
    for index, tmp in enumerate(timg_proj):
        t1 = prj_face[fin]
        t2 = tmp
        e_dist = np.sqrt(np.sum(np.square(prj_face[fin] - tmp), axis=1))
        dist.append(np.min(e_dist))
        if np.min(e_dist) < 12.5*c[fin]:
            pred.append(np.argmin(e_dist).tolist())
        else:
            pred.append(-1)

    pred_imgs = []
    for index, p in enumerate(pred):
        pred_imgs.append(timages[index])
        if p != -1:
            pred_imgs.append(images[p])
        else:
            #pred_imgs.append(-1)
            pred_imgs.append(blnkimg)

    pred_imgs  = np.array(pred_imgs)
    display_images(pred_imgs,timages.shape[0],2)


# task 4
all_com = np.arange(1,images.shape[0])
eg_faceset, prj_face = get_egfaces(all_com)
x_val = []
error = []
for fin, egf in enumerate(eg_faceset):
    timg_proj = np.dot(timg_vec, np.transpose(egf))
    pred = []
    for index, tmp in enumerate(timg_proj):
        e_dist = np.sqrt(np.sum(np.square(prj_face[fin] - tmp), axis=1))
        if np.min(e_dist) < 13.5*all_com[fin]:
            pred.append(1)
        else:
            pred.append(-1)

    x_val.append(all_com[fin])
    error.append((np.sum(tlabels != np.array(pred))/timages.shape[0])*100)


plt.plot(x_val, error)
plt.ylabel("Error rate")
plt.xlabel("k value")
plt.ylim(0,100)
plt.show()


