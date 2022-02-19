# MODEL 1
## dataset: FASHION_MNIST
## net: MLPClassifier (Multi Layer Perceptron Classifier)

# ---------------------------------------------------------------#
#        1 - TREINAR E GUARDAR A REDE NEURONAL                   #
# ---------------------------------------------------------------#

# Importing the main libraries
import pandas as pd

# Importing the train dataset
dataset_train = pd.read_csv('fashion-mnist_train.csv')
X_train = dataset_train.iloc[:, 1:].values
y_train = dataset_train.iloc[:, 0].values

# Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only on training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)

# Training the network
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=True, epsilon=1e-08,
       hidden_layer_sizes=(100,50,25), learning_rate='constant',
       learning_rate_init=0.001, max_iter=1000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

clf.fit(X_train, y_train)

score_train = clf.score(X_train, y_train)
print("Training set score: %.4f" % (score_train*100))

from sklearn.multiclass import OneVsRestClassifier as ovr
classifier_fit = ovr(clf).fit(X_train, y_train)

# Save the Model
from sklearn.externals import joblib
joblib.dump(clf, 'MLP_fit.joblib')
joblib.dump(classifier_fit, 'MLP_class_fit.joblib')

print("Completo. Passar ao step 2.")

# ---------------------------------------------------------------#
#        2 - TESTE DA REDE NEURONAL                              #
# ---------------------------------------------------------------#

# MODEL 1
## dataset: FASHION_MNIST
## net: MLPClassifier (Multi Layer Perceptron Classifier)

### STEP 2 - LOAD AND TEST THE NEURAL NETWORK

### Importing the main libraries
import matplotlib.pyplot as plt
import pandas as pd

### Load the Model
from sklearn.externals import joblib
clf = joblib.load('MLP_fit.joblib')
clf_fit = joblib.load('MLP_class_fit.joblib')

dataset_test = pd.read_csv('fashion-mnist_test.csv')
X_test = dataset_test.iloc[:, 1:].values
y_test = dataset_test.iloc[:, 0].values

### Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_test)
X_test = scaler.transform(X_test)

y_pred = clf.predict(X_test)

## METRICS #####################################
from sklearn import metrics as mt # +classification_report, +confusion_matrix,
                                    # +accuracy, +auc, roc_auc_score, +f1_score,
                                    # precision_recall_curve, -precision_recall_fscore_support
                                    # +precision_score, +recall_score, +roc_curve

class_names = ['t_shirt_top', 'trouser', 'pullover',
                   'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

### Compute confusion matrix
cnf_matrix = mt.confusion_matrix(y_test, y_pred)

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
accuracy = mt.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print("Test set ACCURACY: %.4f" % (accuracy*100))

### Precision:
precision = mt.precision_score(y_test, y_pred,  average='weighted')
print("Test set PRECISION SCORE: %.4f" % (precision*100))

### Recall:
recall = mt.recall_score(y_test, y_pred,  average='weighted')
print("Test set RECALL SCORE: %.4f" % (recall*100))

### F1-Score Mean:
f1_score = mt.f1_score(y_test, y_pred, average='weighted')
print("Test set F1-SCORE: %.4f" % (f1_score*100))

### F-Measure:
f_measure = 2*precision*recall/(precision+recall)
print("Test set F-MEASURE: %.4f" % (f_measure*100))

### Classification Report - Precision, Recall, F1-Score:
crp = mt.classification_report(y_test,y_pred, target_names = class_names, digits=4)
pmaps.plot_classification_report(crp, title='Classification report ', cmap='YlOrBr_r')

y_score = clf_fit.predict_proba(X_test)

### ROC Curve: <<<<<<<<<<<<<<<<<<<<< FAZER PARA CADA CLASSE!!! >>>>>>>>>>>>>>>
for cl in range(0,10):
    fpr, tpr, thresholds = mt.roc_curve(y_test, y_score[:,cl], pos_label=cl)
    ### AUC Score:
    auc = mt.auc(fpr, tpr) # or auc = np.trapz(tpr,fpr)
    plt.title('Receiver Operating Characteristic - {} - {}'.format(cl, class_names[cl]))
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.6f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    # -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.neural_network import MLPClassifier


###################################################################################################
# 3 - SELECIONAR AS MELHORES CARACTERÍSTICAS DO CONJUNTO DE DADOS PARA O MODELO                   #
###################################################################################################

# Criação de uma instância de um classificador binário
classifier = linear_model.LogisticRegression(multi_class='auto', max_iter=200)

# Definição da função objetivo
def f_per_particle(m, alpha):

    """Computes for the objective function per particle

    Inputs
    ------
    m : numpy.ndarray
        Binary mask that can be obtained from BinaryPSO, will
        be used to mask features.
    alpha: float (default is 0.5)
        Constant weight for trading-off classifier performance
        and number of features
        
    Returns
    -------
    numpy.ndarray
        Computed objective function
    """
    total_features = X_train.shape[1]
    
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X_train
    else:
        X_subset = X_train[:,m==1]
        print(X_subset.shape[1])
    # Perform classification and store performance in P
    classifier.fit(X_subset, y_train)
    P = (classifier.predict(X_subset) == y_train).mean()

    # Compute for the objective function
    j = (alpha * (1.0 - P)
        + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))

    return j

#########################

def f(x, alpha=0.88):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """

    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

# Initialize swarm, arbitrary
options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 8, 'p':2}

# Call instance of PSO
dimensions = X_train.shape[1]
#optimizer.reset()
optimizer = ps.discrete.BinaryPSO(n_particles=20, dimensions=dimensions, options=options)

# Perform optimization
cost, px = optimizer.optimize(f, iters=5, verbose=0) #**********

# Testar 2 Modelos: Regressão Logística e Perceptrão Multi-Camada após feature selection:
# Get the selected features from the final positions
X_selected_features_train_set = X_train[:,px==1]
X_selected_features_test_set = X_test[:,px==1]

"""
# Create another instance of LogisticRegression
classfierLR = linear_model.LogisticRegression(multi_class='auto', solver='warn')
# Perform classification on train set
cl = classfierLR.fit(X_selected_features_train_set, trainy)
# Compute performance on test set
subset_performance = (cl.predict(X_selected_features_test_set) == testy).mean()

print('LR: Performance do subconjunto: %.3f' % (subset_performance))
"""

# Create an instance of MLPClassifier
mlpClassifier = MLPClassifier(hidden_layer_sizes = (200,100), early_stopping = False)

# Perform classification on train set
mlp = mlpClassifier.fit(X_selected_features_train_set, y_train)

# Compute performance on test set
y_pred = mlp.predict(X_selected_features_test_set)
subset_performance = (y_pred == y_test).mean()

print('MLP: Performance do subconjunto: %.3f' % (subset_performance))

# Guardar datasets de treino e teste em csv, já normalizado e com as features selecionadas
# Guardar o Dataframe num csv (opcional)
X_selected_features_train_set = pd.DataFrame(X_selected_features_train_set)
y_train = pd.DataFrame(y_train)
dataTreino = pd.concat([X_selected_features_train_set, y_train], axis=1)
dataTreino.to_csv('fashion-mnist_train2.csv', sep=',', header=False, encoding='utf-8')

X_selected_features_test_set = pd.DataFrame(X_selected_features_test_set)
y_test = pd.DataFrame(y_test)
dataTeste = pd.concat([X_selected_features_test_set, y_test], axis=1)
dataTeste.to_csv('fashion-mnist_test2.csv', sep=',', header=False, encoding='utf-8')

#####################################################################################################
#                               4 - RESULTADOS                                                      #
#####################################################################################################

from sklearn import metrics as mt # +classification_report, +confusion_matrix,
                                    # +accuracy, +auc, roc_auc_score, +f1_score,
                                    # precision_recall_curve, -precision_recall_fscore_support
                                    # +precision_score, +recall_score, +roc_curve

class_names = ['t_shirt_top', 'trouser', 'pullover',
                   'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

### Compute confusion matrix
cnf_matrix = mt.confusion_matrix(y_test, y_pred)

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
accuracy = mt.accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
print("Test set ACCURACY: %.4f" % (accuracy*100))

### Precision:
precision = mt.precision_score(y_test, y_pred,  average='weighted')
print("Test set PRECISION SCORE: %.4f" % (precision*100))

### Recall:
recall = mt.recall_score(y_test, y_pred,  average='weighted')
print("Test set RECALL SCORE: %.4f" % (recall*100))

### F1-Score Mean:
f1_score = mt.f1_score(y_test, y_pred, average='weighted')
print("Test set F1-SCORE: %.4f" % (f1_score*100))

### F-Measure:
f_measure = 2*precision*recall/(precision+recall)
print("Test set F-MEASURE: %.4f" % (f_measure*100))

### Classification Report - Precision, Recall, F1-Score:
crp = mt.classification_report(y_test,y_pred, target_names = class_names, digits=4)
pmaps.plot_classification_report(crp, title='Classification report ', cmap='YlOrBr_r')

y_score = clf_fit.predict_proba(X_test)

### ROC Curve: <<<<<<<<<<<<<<<<<<<<< FAZER PARA CADA CLASSE!!! >>>>>>>>>>>>>>>
for cl in range(0,10):
    fpr, tpr, thresholds = mt.roc_curve(y_test, y_score[:,cl], pos_label=cl)
    ### AUC Score:
    auc = mt.auc(fpr, tpr) # or auc = np.trapz(tpr,fpr)
    plt.title('Receiver Operating Characteristic - {} - {}'.format(cl, class_names[cl]))
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.6f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    # -*- coding: utf-8 -*-

print ("Continuar para o ficheiro 'Final'")