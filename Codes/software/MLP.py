import math
import timeit
import glob
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from scipy.fftpack import fft
from statistics import mode
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#from joblib import dump,load
from sklearn.externals import joblib

def dataProcess():
    # Window size is 50 because sampling rate of the sensors from the obtained dataset is 50Hz
    # every 1 second, 50 readings are being taken, so 1 reading takes 1/50 = 0.02 seconds
    # 50% overlap, hence overlap is 25 for window size of 50
    window_size = 50
    overlap = 25
    # dataframe = pd.read_csv('labelled_data2.csv')

    path = r'C:\Users\Siri\Desktop\data1'  # use your path
    all_files = glob.glob(path + "/*.csv")
    list = []
    for filename in all_files:
        df = pd.read_csv(filename)
        list.append(df)
    dataset = pd.concat(list)

    # Drop unnecessary columns from the dataset
    clean_data = dataset.drop(columns=['current', 'voltage', 'power', 'energy'])
    print('Data: ' + str(clean_data.shape[0]) + ' samples, ' + str(clean_data.shape[1]) +
          ' columns.')
    data_array = clean_data.values
    sample_size = math.ceil(len(data_array))
    # Split X and Y data where X data is the inputs from accelerometer and gyroscope, and Y is the class labels
    X_data = data_array[0:sample_size, 1:]  # data taken from only 2nd column onwards
    Y_data = data_array[0:sample_size, 0]  # only 1st column contains class labels
    segmented_data = []
    labeled_data = []

    # There are 758341 rows in the full dataset used.
    # the code implements the sliding window algorithm will split these rows into the segmented data empty array.
    # each row in that segmented data array will be a row of X_data from 0:overlap (so 0:50, 25:75, 50:100).
    # So this results in a total of 30333 rows for both the overlapped and X and corresponding Y data.
    for i in range(int(len(X_data) / overlap)):
        segmented_data.append(X_data[i * overlap:((i * overlap) + window_size), 0:])
        labeled_data.append(Y_data[i * overlap:((i * overlap) + window_size)])

    return labeled_data, segmented_data

def extractFeatures(labeled_data, segmented_data, scaler):
    # for each row of the segmented data array, and then for each X data within a row of the segmented data,
    # we take all 50 rows of 1 column of data (xyz of acc/gyro)
    # then we use these array of values for features in temp_row.
    # Each data column is multiplied by the number of features you have.
    # So if we have 6 columns, and 7 features, its 42 columns of data.
    classes = []
    features = []
    for i in range(len(segmented_data)):
        featureRows = []
        for j in range(0, 24):
            dataRow = segmented_data[i][0:, j]
            min = np.amin(dataRow)
            max = np.amax(dataRow)
            mean = np.mean(dataRow)
            median = np.median(dataRow)
            rms = np.sqrt(np.mean(dataRow ** 2))
            std = np.std(dataRow)
            q75, q25 = np.percentile(dataRow, [75, 25])
            iqr = q75 - q25
            featureRows.append(min)
            featureRows.append(max)
            featureRows.append(mean)
            featureRows.append(median)
            featureRows.append(rms)
            featureRows.append(std)
            featureRows.append(iqr)
            #Frequency Domain Feature - Power Spectral Density
            fourier_temp = fft(dataRow)
            # Freq domain features = Power spectral density, summation |ck|^2
            fourier = np.abs(fourier_temp) ** 2
            density = 0
            for x in range (len(fourier)):
                density = density + (fourier[x] * fourier[x])
            psd = density / len(fourier)
            featureRows.append(psd)

        # features are added to the features array (as a single dimension).
        features.append(featureRows)

    # Same as above sliding window split but for Y data
    # We use mode because we are overlapping the class labels for each window.
    # there might be transition moves in between the windows.
    # so the most frequently occurring label in this window is chosen as the output label for the prediction.
    # In case there is an equal 50:50 split of class labels, then we choose the first label as shown in the ‘except’ block.
    for i in range(len(labeled_data)):
        try:
            classes.append(int(mode(labeled_data[i])))
        except:
            classes.append(int(labeled_data[i][0]))

    # Feature Selection
    # selector = SelectKBest()
    # import f_classif, k=33
    # features = selector.fit_transform(features, classes)

    # Feature Scaling
    # Standardization of features rescales data to have a mean of 0 and a standard deviation of 1 (unit variance).
    # Feed the scaler with the features array, and the transform feature standardizes the features.
    scaler.fit(features)
    features = scaler.transform(features)
    # features2= np.asarray(features)
    # np.savetxt('features.csv', features2, delimiter=',')
    # classes2 = np.asarray(classes)
    # np.savetxt('classes.csv', classes2, delimiter=',')
    # exit()
    return features, classes

#  pd.series prints the classes and the numbers of the classes as actual and predicted.
#  actual and predicted is cross tabbed (which puts them in a cross matrix style).
def printConfusionMatrix(y_actualValues, y_predictedValues):
    y_actual = pd.Series(y_actualValues, name='Actual Class')
    y_predicted = pd.Series(y_predictedValues, name='Predicted Class')
    matrix = pd.crosstab(y_actual, y_predicted, rownames=['Actual Class'], colnames=['Predicted Class'], margins=True)
    print(matrix)

def KfoldValidation(X, y, clf):
    # KFold function that returns the split array (10 splits and shuffles).
    kf = KFold(n_splits=10, shuffle=True)
    fold_index = 0
    sum_accuracy = 0
    # sum_precision = 0
    # sum_recall = 0
    kf.get_n_splits(X)
    # split according to the X data in kf.split(X), kf.split (this function also returns the train and test indexes)
    # train index and text index iterated throughout the X dataset (same for the corresponding y classes).
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y = np.asarray(y)
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        # predictions = clf.predict(X[test_index])
        # matrix = confusion_matrix(y[test_index], predictions)
        accuracy = clf.score(X[test_index], y[test_index])
        print('In %i th fold, accuracy = %f' % (fold_index, accuracy))
        sum_accuracy = sum_accuracy + accuracy
        fold_index = fold_index + 1

    # first row of all features in the X dataset (6 sensor readings * 7 functions = 42 features)
    sample_data = X[0:1, 0:]
    start_time = timeit.default_timer()
    prediction = clf.predict(sample_data)
    total_time = (timeit.default_timer() - start_time) * 1000
    print('One prediction takes ' + str(total_time) + ' ms.')

    #  We then fit the classifier with the train data and run the predictions and get accuracy for each split (0-9).
    return sum_accuracy / fold_index

def main():
    # df_newtest.to_csv('test.csv')
    dataScaler = StandardScaler()
    labelData, segmentData = dataProcess()
    X, y = extractFeatures(labelData, segmentData, dataScaler)
    joblib.dump(dataScaler, 'scaler.pkl', protocol=2)  # Save scaler

    # For RF, number of estimators determines the total runtime and also classification accuracy.
    # Limited to 10 in this dataset for quick training time but can be increased more for increased accuracy
    #  Passing specific random_state=1 (or any other value), then each time you run, you'll get the same result.
    #rf = RandomForestClassifier(n_estimators=20, random_state=1)
    #accuracy = KfoldValidation(X, y, rf)
    #y_predicted = rf.predict(X) #test

    # # rf.fit trains the model with the dataset. We are saving all the data into the model.
    # rf.fit(X, y) #train
    # # dump(rf, 'rf.joblib')
    # print("KFold validation accuracy for Random Forest is {}".format(accuracy))
    # print("\n")
    # print("Classification Report:\n")
    # print(classification_report(y, y_predicted, classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    # print("\n")
    # print("Confusion Matrix:\n")
    #printConfusionMatrix(y, y_predicted)

    # hidden layer(x,y,z). x=no. of input neurons, y=no. of hidden layers, z=no. of connections
    # ‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
    # cannot use relu as relu gives only non-negative values, but we need from -1 to 1 for dance moves
    # early_stopping = True > to terminate training when validation score is not improving.
    # max_iter > Maximum number of iterations.
    # The solver iterates until convergence (determined by ‘tol’) or this number of iterations.
    # shuffle > Whether to shuffle samples in each iteration.
    # batch_size > Size of minibatches for stochastic optimizers.
    # verbose > Whether to print progress messages to stdout.
    # tol > Tolerance for the optimization.
    mlp = MLPClassifier(hidden_layer_sizes=(125, 100, 25), early_stopping=True, max_iter=200, shuffle=True,
                        batch_size=100, activation='tanh', verbose=True, tol=0.001, learning_rate='adaptive')
    accuracy = KfoldValidation(X, y, mlp)
    y_predicted = mlp.predict(X)
    print("KFold validation accuracy for Multilayer Perceptron Classifier is {}".format(accuracy))
    print("\n")
    print("Classification Report:\n")
    print(classification_report(y, y_predicted, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    print("\n")
    print("Confusion Matrix:\n")
    printConfusionMatrix(y, y_predicted)

    mlp2 = MLPClassifier(hidden_layer_sizes=(125, 100, 25), early_stopping=True, max_iter=200, shuffle=True,
                         batch_size=100, activation='tanh', verbose=True, tol=0.001, learning_rate='adaptive')
    mlp2.fit(X, y)
    #dump(mlp,'mlp.joblib')
    # Save MLP Model
    joblib.dump(mlp2, 'mlp.pkl', protocol=2)
    print ("Model Saved\n")

    # Classification Report:
    # precision = ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
    # precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
    # recall = ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.
    # recall is intuitively the ability of the classifier to find all the positive samples.
    # The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall,
    # where an F-beta score reaches its best value at 1 and worst score at 0.
main()
