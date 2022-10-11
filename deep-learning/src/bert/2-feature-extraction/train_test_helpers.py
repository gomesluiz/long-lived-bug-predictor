import numpy as np

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

NUM_PROCESSORS = 16
RANDOM_STATE = 42
KNN = 'knn'
NB = 'nb'
NN = 'nn'
RF = 'rf'
SVM = 'svm'
VC = 'vc'


def build_classifiers(params=None, num_processors=NUM_PROCESSORS):
    """Build a list of classifiers algorithms to be trained
       and evaluated. 

       Returns:
            classifiers: a list of classifier name, classifier 
            object and hyperparameters dictionary. 
    """

    classifiers = []

    knn = KNeighborsClassifier(n_jobs=num_processors)
    nb = GaussianNB()
    svm = SVC(random_state=RANDOM_STATE)
    rf = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=num_processors)
    nn = MLPClassifier(max_iter=2000, random_state=RANDOM_STATE)

    if params is None:
        classifiers.append(
            (KNN,
             knn,
                {'n_neighbors': [5, 11, 15, 21, 25, 33]}
             )
        )        
        classifiers.append(
            (NB,
             nb,
                {'var_smoothing': np.logspace(0, -9, num=100)}
             )
        )
        classifiers.append(
            (NN,
             nn,
                {'hidden_layer_sizes': [(10, 20, 30, 40, 50), (20,)],
                 'activation': ['relu'],
                 'solver': ['adam'],
                 'alpha': [0.0001, 0.05]
                 }
             )
        )
        classifiers.append(
            (RF,
             rf,
                {'max_features': [25, 50, 75, 100]}
             )
        )
        classifiers.append(
            (SVM,
             svm,
                {'kernel': ['rbf'],
                 'gamma': [2**(-15), 2**(-10), 2**(-5), 2**(0), 2**(5)],
                 'C': [2**(-5), 2**(0), 2**(5), 2**(10)]
                 }
             )
        )
    else:
        estimators = [(KNN, knn), (NB, nb), (NN, nn), (SVM, svm), (RF, rf)]
        params = params[['classifier', 'params']].set_index(
            'classifier').to_dict('index')
        for name, estimator in estimators:
            best_params = eval(params[name]['params'])
            estimator.set_params(**best_params)

        classifiers.append(
            (VC, VotingClassifier(estimators=estimators,
                                  voting='hard', n_jobs=num_processors), None)
        )

    return classifiers


def make_model(X, y, model, parameters):

    if parameters is not None:
        model = GridSearchCV(
            estimator=model,
            param_grid=parameters,
            refit=True,
            n_jobs=NUM_PROCESSORS
        )

    model.fit(X, y)
    return model


def score_model(X, y, model):
    kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    scores = cross_val_score(
        model, X, y, cv=kf, scoring='balanced_accuracy', n_jobs=NUM_PROCESSORS)
    return (scores, scores.mean(), scores.std())


def evaluate_model(X, y, model):
    y_hat = model.predict(X)
    acc = model.score(X, y)
    balanced_acc = balanced_accuracy_score(y, y_hat)
    return (acc, balanced_acc, y_hat)


def count_bug_classes(arr):
    """Counts bug classes."""
    return [np.count_nonzero(arr == 0),  np.count_nonzero(arr == 1)]
