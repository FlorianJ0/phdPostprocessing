import numpy as np
from time import time
from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sknn.mlp import Classifier, Layer, Regressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import vapeplot
from termcolor import colored
from scipy import stats, interp
from time import time
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import LocalOutlierFactor
from sklearn import linear_model
import random

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

cmap = vapeplot.cmap('cool')
outliers = 0
nneighbours = 20
randomParamsSearch = 0
eval = 1
rocanal = 0
methodList = ['nn', 'svc', 'nnreg', 'linReg', 'svr']
method = methodList[-1]
dataList = ['local', 'global']
data = dataList[0]


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print('Model with rank: {0}'.format(i))
            print('Mean validation score: {0:.3f} (std: {1:.3f})'.format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print('Parameters: {0}'.format(results['params'][candidate]))
            print('')


def coloring(thresh, val, name):
    if val < thresh:
        print('{} = {}'.format(name, colored(val, 'red')))
    else:
        print('{} = {}'.format(name, colored(val, 'green')))


def score(y_test, y_pred):
    roc = roc_auc_score(np.array(y_test).astype(int), np.array(y_pred))
    cohen = cohen_kappa_score(np.array(y_test).astype(int), np.array(y_pred))
    f1 = f1_score(np.array(y_test).astype(int), np.array(y_pred))
    coloring(0.75, roc, 'roc score')
    coloring(0.25, cohen, 'cohen kappa score')
    coloring(0.5, f1, 'f1 score')
    return


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.RdYlGn, plot=True):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    sns.set()
    sns.set_style('white')
    sns.set_context('talk')
    sns.despine()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print('Normalized confusion matrix')
    # else:
    #     print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if plot:
        plt.show()


if data == 'local':
    localData = np.load('localdata.npy')

    X = localData[:, [4, 5, 6, 9, 10, 11, 15, 16, 25]].astype(float)
    X = X[:10000, :]
    y = localData[:, -4].astype(float)
    y = y[:10000]

    imp.fit(X)
    X = imp.transform(X)
    X = preprocessing.scale(X, axis=0)
    # imp.fit(y)
    # y = imp.transform(y)
    # y = preprocessing.scale(y, axis=0)

elif data == 'global':
    globalData = np.load('globaldata.npy')

    X = globalData[:, 34:49].astype(float)
    y = globalData[:, -2]
    y[y == 'True'] = 1
    y[y == 'False'] = 0
    y = y.astype(int)

    imp.fit(X)
    X = imp.transform(X)
    X = preprocessing.scale(X, axis=0)

if outliers:
    clf = LocalOutlierFactor(n_neighbors=nneighbours)
    y = clf.fit_predict(X)

random_state = random.randint(1e3, 1e6)

if method == 'nn':
    classifier = Pipeline([
        ('neural network', Classifier(layers=[
            Layer('Linear', name='hidden0', units=5),
            Layer('Rectifier', name='hidden1', units=3),
            Layer('Linear', name='hidden2', units=5),
            Layer('Softmax', units=2)],
            learning_rate=0.004095, n_iter=100))])
elif method == 'svc':
    classifier = svm.SVC(kernel='sigmoid', probability=True,
                         random_state=random_state, verbose=True)
elif method == 'svr':
    classifier = svm.SVR(kernel='rbf', verbose=True)

elif method == 'nnreg':
    classifier = Regressor(
        layers=[
            Layer('Linear', name='hidden0', units=10),
            Layer('Sigmoid', name='hidden0', units=10),
            Layer('Tanh', name='hidden0', units=10),
            # Layer('Linear', name='hidden0', units=50),

            # Layer('Rectifier', name='hidden1', units=3),
            # Layer('Linear', name='hidden2', units=5),
            Layer('Linear')],
        learning_rate=0.001,
        n_iter=25)
elif method == 'linReg':
    classifier = linear_model.LinearRegression()
else:
    print('dumbass')
    exit()

if eval:
    if method in ['nnreg', 'linReg','svr']:
        rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=random_state)
        for train, test in rkf.split(X, y):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            y_pred = classifier.fit(X_train, y_train).predict(X_test)
            print "Results of Linear Regression...."
            print "================================\n"
            # The coefficients
            # print('Coefficients: ', classifier.coef_)
            # The mean square error
            print("Residual sum of squares: %.2f"
                  % np.mean((classifier.predict(X_test) - y_test) ** 2))
            # Explained variance score: 1 is perfect prediction
            print('Variance score: %.2f' % classifier.score(X_test, y_test))

            # Plot outputs
        #     plt.scatter(X_test, y_test, color='black')
        #     plt.plot(diabetes_X_test, classifier.predict(X_test), color='blue',
        #              linewidth=3)
        #
        #     plt.xticks(())
        #     plt.yticks(())
        #
        # plt.show()
    else:
        rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=random_state)
        ini = 0
        for train, test in rkf.split(X, y):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            y_pred = classifier.fit(X_train, y_train).predict(X_test)

            # Compute confusion matrix
            cnf_matrix = confusion_matrix(y_test, y_pred)
            np.set_printoptions(precision=2)

            # Plot non-normalized confusion matrix
            class_names = ['T', 'F']

            # Plot normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                                  title='Normalized confusion matrix', cmap=cmap, plot=True)

            score(y_test, y_pred)
            ini += 1

if randomParamsSearch:
    neural = 0
    if neural:
        nn = Classifier(
            layers=[
                Layer('Rectifier', name='hidden0', units=100),
                Layer('Rectifier', name='hidden1', units=100),
                Layer('Rectifier', name='hidden2', units=100),
                Layer('Softmax')],
            learning_rate=0.001,
            n_iter=100)
        params = {
            'learning_rate': stats.uniform(0.0001, 0.01),
            'hidden0__units': stats.randint(2, 12),
            'hidden0__type': ['Rectifier', 'Sigmoid', 'Tanh', 'Linear', 'Softmax', 'ExpLin'],
            'hidden1__units': stats.randint(2, 12),
            'hidden1__type': ['Rectifier', 'Sigmoid', 'Tanh', 'Linear', 'Softmax', 'ExpLin'],
            'hidden2__units': stats.randint(2, 12),
            'hidden2__type': ['Rectifier', 'Sigmoid', 'Tanh', 'Linear', 'Softmax', 'ExpLin'],
        }
    else:
        nn = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state, max_iter=-1)
        params = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }
    n_iter_search = 4

    random_search = RandomizedSearchCV(estimator=nn, param_distributions=params, n_iter=n_iter_search, n_jobs=1,
                                       error_score=0)

    start = time()
    random_search.fit(X, y)
    print('RandomizedSearchCV took %.2f seconds for %d candidates'
          ' parameter settings.' % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

if rocanal:
    # #############################################################################
    # Classification and ROC analysis

    # Run classifier with cross-validation and plot ROC curves
    sns.set()
    sns.set_style('white')
    sns.set_context('paper')
    sns.despine()
    vapeplot.set_palette('vaporwave')

    cv = StratifiedKFold(n_splits=5)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC with ' + method + ' classifier')
    plt.legend(loc="lower right")
    plt.show()
