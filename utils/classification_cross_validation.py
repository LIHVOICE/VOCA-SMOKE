from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

def pipeline_cross_val(names, Classifiers, X, y, kfold, scaler = StandardScaler(), reduction =None, n_components=None, n_features_to_select=10, step=1): 
    #depends on the task:
    #if reduction = 'PCA' -> PCA()
    #if reduciton = 'Feature_selection' -> RFE feature selection""""
    models = zip(names, Classifiers)

    result_acc = []
    result_bacc = []
    result_precision= []
    result_recall= []
    result_f1= []
    result_auc = []

    for _, model in models:

        new_model = Pipeline([('scaler', scaler), ('model', model)])

        if reduction =='PCA':
            new_model =  Pipeline([('scaler', scaler), ('reduction', PCA(random_state=42, n_components=n_components)), ('model', model)])
        elif reduction == 'Feature_selection':
            new_model =  Pipeline([('scaler', scaler), ('reduction', RFE(RandomForestClassifier(random_state=42), n_features_to_select=n_features_to_select, step=step)), ('model', model)])

        
        cv_results_acc = cross_val_score(new_model, X, y, cv = kfold, n_jobs = -1, scoring = 'accuracy')
        result_acc.append("{0:.2f} ({1:.2f})".format(cv_results_acc.mean(), cv_results_acc.std()))

        cv_results_acc = cross_val_score(new_model, X, y, cv = kfold, n_jobs = -1, scoring = 'balanced_accuracy')
        result_bacc.append("{0:.2f} ({1:.2f})".format(cv_results_acc.mean(), cv_results_acc.std()))

        cv_results_precision= cross_val_score(new_model, X, y, cv = kfold, scoring = 'precision')
        result_precision.append("{0:.2f} ({1:.2f})".format(cv_results_precision.mean(), cv_results_precision.std()))

        cv_results_recall = cross_val_score(new_model, X, y, cv = kfold, scoring = 'recall')
        result_recall.append("{0:.2f} ({1:.2f})".format(cv_results_recall.mean(), cv_results_recall.std()))

        cv_results_f1 = cross_val_score(new_model, X, y, cv = kfold, scoring = 'f1')
        result_f1.append("{0:.2f} ({1:.2f})".format(cv_results_f1.mean(), cv_results_f1.std()))

        cv_results_auc = cross_val_score(new_model, X, y, cv = kfold, scoring = 'roc_auc')
        result_auc.append("{0:.2f} ({1:.2f})".format(cv_results_auc.mean(), cv_results_auc.std()))
       
  
    data={'Classifier':names, 'accuracy':result_acc, 'balanced_accuracy':result_bacc, 'Precision':result_precision, 'Recall':result_recall, 'F1':result_f1, 'AUC':result_auc}
    df=pd.DataFrame(data)
    return df
