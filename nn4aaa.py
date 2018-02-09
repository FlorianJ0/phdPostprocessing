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
from sknn.mlp import Classifier, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import vapeplot
from termcolor import colored
from scipy import stats
from time import time

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

cmap = vapeplot.cmap('cool')
# do random search or evaluate
random = 0
eval = 1


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
        print('{} = {:.2}'.format(name, colored(val, 'red')))
    else:
        print('{} = {:.5}'.format(name, colored(val, 'green')))
    print(type(colored(val, 'red')))


def score(y_test, y_pred):
    y_pred[y_pred == 'True'] = 1
    y_pred[y_pred == 'False'] = 0
    y_pred = np.array(list(zip(*y_pred)[0])).astype(int)
    y_test[y_test == 'True'] = 1
    y_test[y_test == 'False'] = 0

    roc = roc_auc_score(np.array(y_test).astype(int), np.array(y_pred))
    cohen = cohen_kappa_score(np.array(y_test).astype(int), np.array(y_pred))
    f1 = f1_score(np.array(y_test).astype(int), np.array(y_pred))
    coloring(0.75, roc, 'roc score')
    coloring(0.25, cohen, 'cohen kappa score')
    coloring(0.5, f1, 'f1 score')

    #  'ROC_score = ', roc_auc_score(np.array(y_test).astype(int), np.array(y_pred))
    # 'cohen_kappa_score = ', cohen_kappa_score(np.array(y_test).astype(int), np.array(y_pred))
    # 'f1_score = ', f1_score(np.array(y_test).astype(int), np.array(y_pred))
    return


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.RdYlGn):
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


localData = np.load('localdata.npy')
globalData = np.load('globaldata.npy')

X = globalData[:, 34:49].astype(float)
y = globalData[:, -2]
imp.fit(X)
X = imp.transform(X)
X = preprocessing.scale(X, axis=0)

random_state = 12883823
testSize = 0.3
if eval:
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=random_state)
    ini = 0
    for train, test in rkf.split(X, y):
        # print('%s %s' % (train, test))
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=testSize, random_state=random_state)

        'available layers: Rectifier, Sigmoid, Tanh, Linear, Softmax, Gaussian, ExpLin'
        pipeline = Pipeline([
            ('neural network', Classifier(layers=[
                Layer('Sigmoid', units=19),
                Layer('Tanh', units=10),
                Layer('Sigmoid', units=14),
                Layer('Softmax', units=2)],
                learning_rate=0.001, n_iter=25, normalize=True))])
        # pipeline.fit(X_train, y_train)
        y_pred = pipeline.fit(X_train, y_train).predict(X_test)
        # cm = confusion_matrix(y_test, y_pred)
        # sns.heatmap(cm, cmap=cmap)

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        class_names = ['T', 'F']

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix', cmap=cmap)
        # print y_pred
        # print 'cohen_kappa_score=== ', cohen_kappa_score(y_test, y_pred.astype(bool))
        # print 'f1_score=== ', f1_score(y_test, y_pred)

        score(y_test, y_pred)

        ini += 1

# plt.show()
if random:
    nn = Classifier(
        layers=[
            Layer('Rectifier', name='hidden0', units=100),
            Layer('Rectifier', name='hidden1', units=100),
            Layer('Rectifier', name='hidden2', units=100),
            Layer('Softmax')],
        learning_rate=0.001,
        n_iter=25)
    params = {
        'learning_rate': stats.uniform(0.0001, 0.05),
        'hidden0__units': stats.randint(2, 22),
        'hidden0__type': ['Rectifier', 'Sigmoid', 'Tanh', 'Linear', 'Softmax', 'ExpLin'],
        'hidden1__units': stats.randint(2, 22),
        'hidden1__type': ['Rectifier', 'Sigmoid', 'Tanh', 'Linear', 'Softmax', 'ExpLin'],
        'hidden2__units': stats.randint(2, 22),
        'hidden2__type': ['Rectifier', 'Sigmoid', 'Tanh', 'Linear', 'Softmax', 'ExpLin'],
    }
    n_iter_search = 4

    random_search = RandomizedSearchCV(nn, param_distributions=params, n_iter=n_iter_search, n_jobs=1, error_score=0)

    # random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
    # n_iter=n_iter_search)

    start = time()
    random_search.fit(X, y)
    print('RandomizedSearchCV took %.2f seconds for %d candidates'
          ' parameter settings.' % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

'''

# exit()
# nn = Classifier(
#     layers=[Layer('Softmax')],
#     learning_rate=0.001,
#     n_iter=25)
pipeline.fit(X_train, y_train)
y_example = pipeline.predict(X_test)
exit()
# build a classifier
clf = RandomForestClassifier(n_estimators=20)
cff = RandomForestClassifier(n_estimators=10)


# Utility function to report best scores
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


# specify parameters and distributions to sample from
param_dist = {'max_depth': [3, None],
              'max_features': sp_randint(1, 11),
              'min_samples_split': sp_randint(2, 11),
              'min_samples_leaf': sp_randint(1, 11),
              'bootstrap': [True, False],
              'criterion': ['gini', 'entropy']}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, n_jobs = 1)

start = time()
random_search.fit(X, y)
print('RandomizedSearchCV took %.2f seconds for %d candidates'
      ' parameter settings.' % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {'max_depth': [3, None],
              'max_features': [1, 3, 10],
              'min_samples_split': [2, 3, 10],
              'min_samples_leaf': [1, 3, 10],
              'bootstrap': [True, False],
              'criterion': ['gini', 'entropy']}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs = 4)
start = time()
grid_search.fit(X, y)

print('GridSearchCV took %.2f seconds for %d candidate parameter settings.'
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
'''
