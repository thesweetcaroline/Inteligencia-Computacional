#####################################################################################################
#      5 -  APLICAR O CAT SWARM OPTIMIZATION COMO OTIMIZADOR DOS PARÃMETROS DA REDE NEURONAL        #                                                   #
#####################################################################################################

import numpy as np
import pandas as pd

dataTrain = pd.read_csv('fashion-mnist_train2.csv', header= None)
dataTest = pd.read_csv('fashion-mnist_test2.csv', header= None)

X_selected_features_train_set = dataTrain.iloc[:,:-1]
y_train = dataTrain.iloc[:,-1]

X_selected_features_test_set = dataTest.iloc[:,:-1]
y_test = dataTest.iloc[:,-1]

class MultiLayerPerceptron:
    def __init__(self, shape, weights=None):
        self.shape = shape
        self.num_layers = len(shape)
        if weights is None:
            self.weights = []
            for i in range(self.num_layers-1):
                W = np.random.uniform(size=(self.shape[i+1], self.shape[i] + 1))
                self.weights.append(W)
        else:
            self.weights = weights


    def run(self, data):
        layer = data.T
        for i in range(0,self.num_layers-1):
            prev_layer = np.insert(layer, 0, 1, axis=0)
            o = np.dot(self.weights[i], prev_layer)
            layer = 1.0 / (1 + np.exp(-o))
        return layer
    
if __name__ == '__main__':
    import pso
    import functools
    import sklearn.metrics
    import sklearn.datasets
    import sklearn.model_selection

    def dim_weights(shape):
        dim = 0
        for i in range(len(shape)-1):
            dim = dim + (shape[i] + 1) * shape[i+1]
        return dim

    def weights_to_vector(weights):
        w = np.asarray([])
        for i in range(len(weights)):
            v = weights[i].flatten()
            w = np.append(w, v)
        return w

    def vector_to_weights(vector, shape):
        weights = []
        idx = 0
        for i in range(len(shape)-1):
            r = shape[i+1]
            c = shape[i] + 1
            idx_min = idx
            idx_max = idx + r*c
            W = vector[idx_min:idx_max].reshape(r,c)
            weights.append(W)
        return weights

    def eval_neural_network(weights, shape, X, y):
        mse = np.asarray([])
        for w in weights:
            weights = vector_to_weights(w, shape)
            nn = MultiLayerPerceptron(shape, weights=weights)
            y_pred = nn.run(X)
            mse = np.append(mse, sklearn.metrics.mean_squared_error(np.atleast_2d(y), y_pred))
        return mse

    num_classes = 10

    X = np.asarray(X_selected_features_train_set)
    X_test = np.asarray(X_selected_features_test_set)
    y = np.asarray(y_train)
    y_test = np.asarray(y_test)
    
    num_inputs = X.shape[1]

    y_true = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        y_true[i, y[i]-1] = 1

    y_test_true = np.zeros((len(y_test), num_classes))
    for i in range(len(y_test)):
        y_test_true[i, y_test[i]-1] = 1

    # Set up
    shape = (num_inputs, 25, 25, num_classes)
    cost_func = functools.partial(eval_neural_network, shape=shape, X=X, y=y_true.T)

    swarm = pso.ParticleSwarm(cost_func, dim=dim_weights(shape), size=150)

    # Train...
    i = 0
    best_scores = [(i, swarm.best_score)]
    print (best_scores[-1])
    while swarm.best_score > 1e-6 and i<750:
    #while i < 1:
        swarm.update()
        i = i+1
        if swarm.best_score < best_scores[-1][1]:
            best_scores.append((i, swarm.best_score))
            print (best_scores[-1])

    # Test...
    best_weights = vector_to_weights(swarm.g, shape)
    best_nn = MultiLayerPerceptron(shape, weights=best_weights)
    y_test_pred = np.round(best_nn.run(X_test))
    print (sklearn.metrics.classification_report(y_test_true, y_test_pred.T))
    accuracy = sklearn.metrics.accuracy_score(y_test_true, y_test_pred.T)
    

from sklearn import metrics as mt

class_names = ['t_shirt_top', 'trouser', 'pullover',
                   'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

### Compute confusion matrix
#cnf_matrix = mt.confusion_matrix(testy.values.argmax(axis=1), y_pred.argmax(axis=1))
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
yy_true = (y_test_true.T==1).astype(np.int).T
yy_true=np.asarray(np.where(yy_true))[1,:]+1

yy_pred =(y_test_pred.T==1).astype(np.int)

y_special=np.asarray(np.where(yy_pred==1)).T[:,-1]+1

cnf_matrix = mt.confusion_matrix(yy_true, y_special)
### Plot non-normalized confusion matrix
import plotMAPS as pmaps
plt.figure(figsize=(10,10))
pmaps.plot_confusion_matrix(cnf_matrix, classes=class_names,
                            title='Confusion matrix, without normalization')
plt.show()

### Plot normalized confusion matrix
plt.figure(figsize=(10,10))
pmaps.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                            title='Normalized confusion matrix')
plt.show()

#### Accuracy:
accuracy = mt.accuracy_score(yy_true, y_special, normalize=True, sample_weight=None)
print("Test set ACCURACY: %.4f" % (accuracy*100))

### Precision:
precision = mt.precision_score(yy_true, y_special,  average='weighted')
print("Test set PRECISION SCORE: %.4f" % (precision*100))

### Recall:
recall = mt.recall_score(yy_true, y_special,  average='weighted')
print("Test set RECALL SCORE: %.4f" % (recall*100))

### F1-Score Mean:
f1_score = mt.f1_score(yy_true, y_special, average='weighted')
print("Test set F1-SCORE: %.4f" % (f1_score*100))

### F-Measure:
f_measure = 2*precision*recall/(precision+recall)
print("Test set F-MEASURE: %.4f" % (f_measure*100))

### Classification Report - Precision, Recall, F1-Score:
crp = mt.classification_report(yy_true,y_special, target_names = class_names, digits=4)
pmaps.plot_classification_report(crp, title='Classification report ', cmap='YlOrBr_r')

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

# Binarize the output
y_train = label_binarize(y_train, classes=[0,1,2,3,4,5,6,7,8,9])
y_test = label_binarize(y_test, classes=[0,1,2,3,4,5,6,7,8,9])
n_classes = y_train.shape[1]

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
y_score = classifier.fit(X_selected_features_train_set, y_train).decision_function(X_selected_features_test_set)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0,10):
    fpr[i], tpr[i], _ = mt.roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = mt.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = mt.roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = mt.auc(fpr["micro"], tpr["micro"])

for i in range(0,10):
    plt.figure()
    lw = 1
    plt.plot(fpr[i], tpr[i], color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - {} - {}'.format(i+1, class_names[i]))
    plt.legend(loc="lower right")
    plt.show()