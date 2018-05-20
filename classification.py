# Sameer Shivpure
# 1001417543

import numpy as np
import sklearn as skl
from sklearn import svm


class Classification:

    def __init__(self):
        choice = input("Please select the dataset from below\n 1 - ATNT Face Images\n 2 - Hand Written Letters\n")

        if choice == '1':
            datafile = "ATNTFaceImages400.txt"
        elif choice == '2':
            datafile = "HandWrittenLetters.txt"
        else:
            print("Choose correct option\n")
            return

        self.k = int(input("Please enter the 'k' value for kNN classifier -\n"))
        self.data = np.loadtxt(datafile, delimiter=",")
        self.dataX = np.transpose(self.data[1:, :])
        self.dataY = self.data[0, :]
        self.k_spilts = 5
        self.kf = skl.model_selection.StratifiedKFold(n_splits=self.k_spilts, random_state=5)
        self.kf.get_n_splits(self.dataX, self.dataY)

        #self.split_dataset()
        # self.k_means()
        self.lin_reg()
        self.knn()
        self.centroid()
        self.svmclf()

    def lin_reg(self):

        print("\nLinear regression classification-\n")
        i =1
        totalacc = 0
        for trIndex, teIndex in self.kf.split(self.dataX, self.dataY):
            trX = self.dataX[trIndex]
            trY = self.dataY[trIndex]
            teX = self.dataX[teIndex]
            teY = self.dataY[teIndex]

            trClass = np.unique(trY)
            if 0 not in trClass:
                trY -= 1

            y = np.zeros((trY.shape[0],trClass.shape[0]))
            y[np.arange(trY.shape[0]),np.array(trY, dtype=np.int32)] = 1
            weight = np.dot(np.linalg.pinv(trX),y)
            predY = np.dot(teX, weight)

            predY = np.argmax(predY, axis=1)
            if 0 not in trClass:
                predY += 1

            accuracy = np.sum(teY == predY.reshape((-1,)))/ len(teY)
            totalacc += accuracy
            print("K = %d, accuracy = %3.2f %%" % (i, (accuracy * 100)))
            i += 1
        print("Average accuracy = %3.2f %%"% ((totalacc/self.k_spilts)*100))
        return

    def knn(self):

        print("\nkNN classification with k = %d -\n" %(self.k))
        i = 1
        totalacc = 0

        for trIndex, teIndex in self.kf.split(self.dataX, self.dataY):
            trX = self.dataX[trIndex]
            trY = self.dataY[trIndex]
            teX = self.dataX[teIndex]
            teY = self.dataY[teIndex]
            pred = []

            for index in range(teX.shape[0]):
                dist = np.linalg.norm(trX - teX[index, :].reshape((1,-1)), axis=1)
                min_indexs = np.argsort(dist)[:self.k]
                predY, ct = np.unique(trY[min_indexs],return_counts=True)
                pred.append(predY[np.argmax(ct)])

            accuracy = np.sum(teY == np.array(pred))/len(teY)
            totalacc += accuracy
            print("K = %d, accuracy = %3.2f %%" % (i, (accuracy * 100)))
            i += 1
        print("Average accuracy = %3.2f %%" % ((totalacc / self.k_spilts) * 100))
        return

    def centroid(self):

        print("\nCentroid classification -\n")
        i = 1
        totalacc = 0

        for trIndex, teIndex in self.kf.split(self.dataX, self.dataY):
            trX = self.dataX[trIndex]
            trY = self.dataY[trIndex]
            teX = self.dataX[teIndex]
            teY = self.dataY[teIndex]

            trClass = np.unique(trY)
            clCentroids = []
            pred = []
            for cl in trClass:
                clCentroids.append(np.mean(trX[np.where(trY == cl)[0], :],axis=0))

            for index in range(teX.shape[0]):
                pred.append(trClass[np.argmin(np.linalg.norm(clCentroids - teX[index, :].reshape((1, -1)), axis=1))])

            accuracy = np.sum(teY == np.array(pred))/len(teY)
            totalacc += accuracy
            print("K = %d, accuracy = %3.2f %%" % (i, (accuracy * 100)))
            i += 1
        print("Average accuracy = %3.2f %%" % ((totalacc / self.k_spilts) * 100))
        return

    def svmclf(self):

        clf = svm.SVC(kernel='linear', C=1, gamma=1)
        print("\nSVM classification -\n")
        i = 1
        totalacc = 0

        for trIndex, teIndex in self.kf.split(self.dataX, self.dataY):
            trX = self.dataX[trIndex]
            trY = self.dataY[trIndex]
            teX = self.dataX[teIndex]
            teY = self.dataY[teIndex]

            clf.fit(trX, trY)
            pred = clf.predict(teX)
            accuracy = np.sum(teY == np.array(pred))/len(teY)
            totalacc += accuracy
            print("K = %d, accuracy = %3.2f %%" % (i, (accuracy * 100)))
            i += 1
        print("Average accuracy = %3.2f %%" % ((totalacc / self.k_spilts) * 100))
        return

    def k_means(self):

        centroids = self.dataX[np.random.randint(0, self.dataX.shape[0], size=self.k),:]

        while True:
            clusters = []
            old_centroids = centroids.copy()

            for data in self.dataX:
                clusters.append(np.argmin(np.linalg.norm(centroids - data.reshape((1,-1)), axis=1)))

            for k in range(self.k):
                centroids[k,:] = np.mean(self.dataX[np.array(clusters) == k,:], axis=0)

            if np.all(old_centroids == centroids):
                self.cluster_means = centroids
                self.pred_labels = clusters
                break

        # accuracy = np.sum(self.dataY == np.array(clusters)) / len(self.dataY)
        # print("K = %d, accuracy = %3.2f %%" % (i, (accuracy * 100)))
        return

    def split_dataset(self):

        classes = np.unique(self.dataY)
        trainX = []
        trainY = []
        testX = []
        testY = []

        for cl in classes:
            ind = np.where(self.dataY == cl)
            trainX.append(self.dataX[ind][:9])
            trainY.append(self.dataY[ind][:9])
            testX.append(self.dataX[ind][9:])
            testY.append(self.dataY[ind][9:])

        self.trainX = np.array(trainX).reshape((-1, self.dataX.shape[1]))
        self.trainY = np.array(trainY).reshape((-1,))
        self.testX = np.array(testX).reshape((-1, self.dataX.shape[1]))
        self.testY = np.array(testY).reshape((-1,))
        return


if __name__ == "__main__":
    Classification()
