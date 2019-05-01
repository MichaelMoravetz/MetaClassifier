from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.base import clone
import numpy as np
from scipy import sparse
import time

class MetaClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):

    """
    A model stacking classifier for sklearn classifiers. Uses Sklearn API to fit and predict, 
        can be used with PipeLine and other sklearn estimators. Must be passed primary list of estimator(s)
        and secondary(meta) classifier. Secondary model trains a predicts on primary level estimators.

    Parameters:
    _-_-_-_-_-_-_-_-_-

    classifiers : {array-like} shape = [n_estimators]
        list of instantiated sklearn estimators.
    meta_classifier : instatiated sklearn estimator.
        This is the secondary estimator that makes the final prediction based on predicted values
        of classifiers.
    use_probability : bool, (default=False) If True calling fit will train meta_classifier on the predicted probabilities
        instead of predicted class labels.
    double_down : bool, (default=False) If True, calling fit will train meta_classifier on both the primary 
        classifiers predicted lables and the original dataset. Otherwise meta_classifier will only be 
        trained on primary classifier's predicted labels.
    average_probability : bool, (default = False) If True, calling fit will fit the meta_classifier with averaged 
        the probabalities from primiary predictions.
    clones : bool, (default = True), If True, calling fit will fit deep copies of classifiers and meta classifier 
        leaving the original estimators unmodified. False will fit the passed in classifiers directly.  This param 
        is for use with non-sklearn estimators who cannot are not compatible with being cloned.  This may be unecesary
        but I read enough things about it not working to set it as an option for safe measure. It is best to clone.
    verbose : int, (0-2) Sets verbosity level for output while fitting.
    

    Attributes:
    _-_-_-_-_-_-_-_-

    clfs_ : list, fitted classifers (primary classifiers)
    meta_clf_ : estimator, (secondary classifier)
    meta_features_ : predictions from primary classifiers

    Methods:
    _-_-_-_-_-_-_-_-
    fit(X, y, sample_weight=None): fit entire ensemble with training data, including fitting meta_classifier with meta_data
            params: (See sklearns fit model for any estimator)
                    X : {array-like}, shape = [n_samples, n_features]
                    y : {array-like}, shape =[n_samples]
                    sample_weight : array-like, shape = [n_samples], optional
    fit_transform(X, y=None, fit_params) : Refer to Sklearn docs
    predict(X) : Predict labels
    get_params(params) : get classifier parameters, refer to sklearn class docs
    set_params(params) : set classifier parameters, mostly used internally, can be used to set parameters, refer to sklearn docs.
    score(X, y, sample_weight=None): Get accuracy score
    predict_meta(X): predict meta_features, primarily used to train meta_classifier, but can be used for base ensemeble performance
    predict_probs(X) : Predict label probabilities for X.

    EXAMPLE******************************************EXAMPLE*******************************************EXAMPLE
    EXAMPLE:          # Instantiate classifier objects for base ensemble
                >>>>  xgb = XGBClassifier()   
                >>>>  svc = svm.SVC()
                >>>>  gbc = GradientBoostingClassifier()
                
                      # Store estimators in list
                >>>>  classifiers = [xgb, svc, gbc]  

                    # Instantiate meta_classifier for making final predictions
                >>>>  meta_classifier = LogisticRegression()

                    # instantiate MetaClassifer object and pass classifiers and meta_classifier
                    # Fit model with training data
                >>>>  clf = Metaclassifier(classifiers=classifiers, meta_classifier=meta_classifier)
                >>>>  clf.fit(X_train, y_train)

                    # Check accuracy scores, predict away...
                >>>>  print(f"MetaClassifier Accuracy Score: {clf.score(X_test, y_test)}  Get it!")
                >>>>  clf.predict(X)
                ---------------------------------------------------------------------------

                fitting 3 classifiers...
                fitting 1/3 classifers...
                ...
                fitting meta_classifier...

                time elapsed: 6.66 minutes
                MetaClassifier Accuracy Score: 99.9   Get it!
    8***********************************************************************************************>
    """


    def __init__(self, classifiers=None, meta_classifier=None, 
                 use_probability=False, double_down=False, 
                 average_probs=False, clones=True, verbose=2):

        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.use_probability = use_probability
        self.double_down = double_down
        self.average_probs = average_probs
        self.clones = clones
        self.verbose = verbose
        


    def fit(self, X, y, sample_weight=None):

        """
        Fit base classifiers with data and meta-classifier with predicted data from base classifiers.
        
        Parameters:
        .-.-.-.-.-.-.-.-.-.-.
        X : {array-like}, shape =[n_samples, n_features]
            Training data m number of samples and number of features
        y : {array-like}, shape = [n_samples] or [n_samples, n_outputs]
            Target feature values.
        
        Returns:
        .-.-.-.-.-.-.-.-.-.-.
        
        self : object, 
            Fitted MetaClassifier
            
          """
        start = time.time()

        # Make clones of classifiers and meta classifiers to preserve original 
        if self.clones:
            self.clfs_ = clone(self.classifiers)
            self.meta_clf_ = clone(self.meta_classifier)
        else:
            self.clfs_ = self.classifiers
            self.meta_clf_ = self.meta_classifier
        
        if self.verbose > 0:
            print('Fitting %d classifiers' % (len(self.classifiers)))

        # Count for printing classifier count
        n = 1

        for clf in self.clfs_:

            if self.verbose > 1:
                print(f"Fitting classifier {n}/{len(self.clfs_)}")
                n +=1

            if sample_weight is None:
                clf.fit(X ,y)
            else:
                clf.fit(X, y, sample_weight)

        # Get meta_features to fit MetaClassifer   
        meta_features = self.predict_meta(X)

        if verbose >1:
            print("Fitting meta-classifier to meta_features")

        # Assess if X is sparse or not and stack horizontally
        elif sparse.issparse(X):
            meta_features = sparse.hstack((X, meta_features))
        else:
            meta_features = np.hstack((X, meta_features))
        
        # Set attribute
        self.meta_features_ = meta_features

        # Check for sample_weight and fit MetaClassifer to meta_features
        if sample_weight is None:
            self.meta_clf_.fit(meta_features, y)
        else:
            self.meta_clf_.fit(meta_features, y, sample_weight=sample_weight)

        stop = time.time()

        if verbose > 0:
            print(f"Estimators Fit! Time Elapsed: {(stop-start)/60} minutes")
            print("8****************************************>")

        return self



    def predict_meta(self, X):
        
        """
        Predicts on base estimators to get meta_features for MetaClassifier.
        
        Parameters:
        -.-.-.-.-.-.-.-.-
        X : np.array, shape=[n_samples, n_features]

        Returns:
        -.-.-.-.-.-.-.-.-
        meta_features : np.array, shape=[n_samples, n_classifiers]
            the 'new X' for the MetaClassifier to predict with.
        
        """
        # Check parameters and run approriate prediction
        if self.use_probability:

            probs = np.asarray([clf.predict_probs(X) for clf in self.clfs_])

            if self.average_probs:
                preds = np.average(probs, axis=0)

            else:
                preds = np.concatenate(probs, axis=1)

        else:
            preds = np.column_stack([clf.predict(X) for clf in self.clfs_])
        
        return preds

    def predict_probs(self, X):

        """
        Predict probabilities for X
        
        Parameters:
        -.-.-.-.-.-.-.-.-
        X : np.array, shape=[n_samples, n_features]

        Returns:
        -.-.-.-.-.-.-.-.-
        probabilities : array-like,  shape = [n_samples, n_classes] 

        """

        meta_features = self.predict_meta(X)

        if self.double_down == False:
            return self.meta_clf_.predict_probs(meta_features)
        
        elif sparse.issparse(X):
            return self.meta_clf_.predict_probs(sparse.hstack((X, meta_features)))

        else:
            return self.meta_clf_.predict_probs(np.hstack((X, meta_features)))


    def predict(self, X):

        """
        Predicts target values. 
        
        Parameters:
        -.-.-.-.-.-.-.-.-
        X : np.array, shape=[n_samples, n_features]

        Returns:
        -.-.-.-.-.-.-.-.-
        predicted labels : array-like,  shape = [n_samples] or [n_samples, n_outputs] 

        """

        meta_features = self.predict_meta(X)

        if self.double_down == False:
            return self.meta_clf_.predict(meta_features)

        elif sparse.issparse(X):
            return self.meta_clf_.predict(sparse.hstack((X, meta_features)))

        else:
            return self.meta_clf_.predict(np.hstack((X, meta_features)))       
