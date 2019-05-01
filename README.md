# MetaClassifier

This class allows multiple classifiers to be stacked. It must be instantiated with both a set of base classifier(s) and a secondary "meta-classifier".  When fit is called on a MetaClassifier object, the data is fit to the base classifiers. Next, predict is invoked on the base classifiers, the predicted results are then passed to meta-classifier and it is fitted to the new "meta features". In this way, the meta-classifer is trained on the predictions of the base classifiers.  
