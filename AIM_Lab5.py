import numpy
import pandas
import matplotlib.pyplot as pyplot
import pylab
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

#loading datasets
train_df = pandas.read_csv('Video_games_esrb_rating.csv')
test_df = pandas.read_csv('test_esrb.csv')

#preprocessing
def preprocessing(df):
    df = df.copy()
    df = df.drop('title', axis = 1)
    X = df.drop('esrb_rating', axis = 1)
    y = df['esrb_rating']
    X = pandas.DataFrame(X, index = X.index, columns = X.columns)
    return X, y

#scaling
scaler = StandardScaler()
X_train, y_train = preprocessing(train_df)
X_train = scaler.fit_transform(X_train)
X_test, y_test = preprocessing(test_df)
X_test = scaler.fit_transform(X_test)

#run classifiers
gp = GaussianProcessClassifier(kernel = None, random_state = 0).fit(X_train, y_train)
gp_predictions = pandas.Series(gp.predict(X_test))
print('GP Classifier accuracy: ' + str(gp.score(X_test, y_test)*100) + '%')

svm = SVC(kernel = 'rbf').fit(X_train, y_train)
svm_predictions = pandas.Series(svm.predict(X_test))
print('SVM Classifier accuracy: ' + str(svm.score(X_test, y_test)*100) + '%')

mlp = MLPClassifier(random_state = 1, max_iter = 300).fit(X_train, y_train)
mlp_predictions = pandas.Series(mlp.predict(X_test))
print('MLP Classifier accuracy: ' + str(mlp.score(X_test, y_test)*100) + '%')

#output pie charts
pylab.subplot(2, 1, 1)
pyplot.pie(y_test.value_counts().sort_index(), labels = sorted(y_test.unique()), autopct='%1.1f%%')
pyplot.title('Original data')
pylab.subplot(2, 1, 2)
pyplot.pie(gp_predictions.value_counts().sort_index(), labels = sorted(gp_predictions.unique()), autopct='%1.1f%%')
pyplot.title('GP Predictions')
pyplot.show()

pylab.subplot(2, 1, 1)
pyplot.pie(y_test.value_counts().sort_index(), labels = sorted(y_test.unique()), autopct='%1.1f%%')
pyplot.title('Original data')
pylab.subplot(2, 1, 2)
pyplot.pie(svm_predictions.value_counts().sort_index(), labels = sorted(svm_predictions.unique()), autopct='%1.1f%%')
pyplot.title('SVM Predictions')
pyplot.show()

pylab.subplot(2, 1, 1)
pyplot.pie(y_test.value_counts().sort_index(), labels = sorted(y_test.unique()), autopct='%1.1f%%')
pyplot.title('Original data')
pylab.subplot(2, 1, 2)
pyplot.pie(mlp_predictions.value_counts().sort_index(), labels = sorted(mlp_predictions.unique()), autopct='%1.1f%%')
pyplot.title('MLP Predictions')
pyplot.show()