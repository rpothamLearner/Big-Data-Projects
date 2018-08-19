import utils
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, ShuffleSplit
from numpy import mean
from sklearn.metrics import *
#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT
RANDOM_STATE = 545510477


'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
    X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
    #X_train, Y_train = utils.get_data_from_svmlight("C://Users//rpothams//Downloads//LN//Big Data//homework1//deliverables//features_svmlight.train") 
    filepath = '../data/test/'
    events_test = pd.read_csv(filepath + 'events.csv')
    feature_map_test = pd.read_csv(filepath + 'event_feature_map.csv')    
    # using modified etl.py methods to construct X_test
    patient_features = create_features(events_test, feature_map_test)
    save_svmlight(patient_features)   
    X_test, _ = utils.get_data_from_svmlight("../data/test/features_svmlight.test")
    return X_train,Y_train,X_test

def aggregate_events(events_df, feature_map_df):
    filtered_events_df = events_df
    filtered_events_df = filtered_events_df[filtered_events_df.value.notnull()]
    filtered_events_df = filtered_events_df[['patient_id', 'event_id', 'value']].groupby(['patient_id', 'event_id']).agg({'value':'sum', 'event_id': 'count'})
    filtered_events_df = filtered_events_df.rename(columns = {'event_id' : 'count', 'value' : 'sum'})
    filtered_events_df.reset_index(inplace = True)
    filtered_events_df['event_code'] = filtered_events_df.event_id.str.extract('([a-zA-Z]+)', expand = False)
    filtered_events_df['value'] = pd.DataFrame([filtered_events_df['count'][i] if filtered_events_df['event_code'][i] == 'LAB' else filtered_events_df['count'][i] for i in range(len(filtered_events_df))])
    filtered_events_df = pd.merge(filtered_events_df, feature_map_df, on = 'event_id')
    filtered_events_df = filtered_events_df[['patient_id', 'event_id', 'idx', 'value']]

    filtered_events_minmax = filtered_events_df.groupby(['event_id'], as_index = False).agg({'value':['min', 'max']})
    filtered_events_minmax.columns = ['_'.join(col).strip() for col in filtered_events_minmax.columns.values]
    filtered_events_minmax = filtered_events_minmax.rename(columns = {'event_id_' : 'event_id'})
    
    aggregated_events = pd.merge(filtered_events_df, filtered_events_minmax, on = 'event_id')
    aggregated_events['value'] = aggregated_events['value']/aggregated_events['value_max']
    aggregated_events = aggregated_events[['patient_id','idx', 'value']]
    aggregated_events = aggregated_events.rename(columns = {'patient_id' : 'patient_id', 'idx' : 'feature_id', 'value' : 'feature_value'})
    return aggregated_events

def create_features(events_df, feature_map_df):
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(events_df, feature_map_df)
    aggregated_events['value'] = list(zip(aggregated_events.feature_id, aggregated_events.feature_value))
    patient_features = aggregated_events.groupby('patient_id', as_index = False)['value'].agg({'value' : lambda value: list(value)})              
    patient_features = patient_features.set_index('patient_id')['value'].to_dict()
    return patient_features

def save_svmlight(patient_features):
    deliverable1 = open('../data/test/features_svmlight.test', 'wb')
    deliverable2 = open('../deliverables/test_features.txt', 'wb')

    for key in patient_features:
        cur_line = '0 '
        #sorted_pt_features = sorted(patient_features[key], key = lambda x:x[0])
        patient_features[key].sort(key=lambda tup: tup[0])  # sorts in place
        for value in patient_features[key]:
            cur_line += str(int(value[0])) + ':' + str("{:.3f}".format(value[1])) + ' '
        deliverable1.write(cur_line)
        deliverable1.write('\n')
    
    for key in patient_features:
        cur_line = str(int(key)) + ' '
        patient_features[key].sort(key=lambda tup: tup[0])  # sorts in place
        for value in patient_features[key]:
            cur_line += str(int(value[0])) + ':' + str("{:.3f}".format(value[1])) + ' '
        deliverable2.write(cur_line)
        deliverable2.write('\n') 

 
'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
    svc = LinearSVC(C = 1.5, loss='squared_hinge', max_iter=1000, penalty='l2', dual = True)
    '''
    #without  feature selection
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    '''
    sfm = SelectFromModel(svc, threshold = 0.15)
    '''
    # DTree for feature selection
    sfm = SelectFromModel(DecisionTreeClassifier(max_depth = 8), threshold=0.01)
    '''
    sfm.fit(X_train, Y_train)
    n_features = sfm.transform(X_train).shape[1]
    print("number of features")
    print(n_features)
    X_transform = sfm.transform(X_train)  
    X_test_transform = sfm.transform(X_test)
    n_estim = 50 #500
    
    '''
    ensemble1 = AdaBoostClassifier(n_estimators = n_estim, random_state = RANDOM_STATE)
    ensemble1.fit(X_transform, Y_train)
    Y_pred = ensemble1.predict(X_test_transform)
    #ensemble1.fit(X_train, Y_train)
    #Y_pred = ensemble1.predict(X_test)
    '''
    ensemble2 = RandomForestClassifier(n_estimators = n_estim, max_depth=15, random_state = RANDOM_STATE)
    ensemble2.fit(X_transform, Y_train)
    Y_pred = ensemble2.predict(X_test_transform)

    '''
    ensemble3 = DecisionTreeClassifier(max_depth=5, random_state = RANDOM_STATE)
    ensemble3.fit(X_transform, Y_train)
    Y_pred = ensemble3.predict(X_test_transform)
    '''
#    GRID_SEARCH
#    clf_pipe = Pipeline(steps = [('lin_svc', sfm), ('adb', Ada_Boost)])
#    parameters = {'lin_svc__estimator__C' : [0.01, 10], 'lin_svc__threshold' : [0.01, 1], 'adb__n_estimators' : [10, 100]}
#  
#    clf_optim = GridSearchCV(clf_pipe, parameters, n_jobs = 10, scoring = 'roc_auc')
#    clf_optim.fit(X_train, Y_train)
#    print(clf_optim.best_params_)
#    for params, mean_score, scores in clf_optim.grid_scores_:
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean_score, scores.std() * 2, params))
#    Y_pred = clf_optim.predict(X_test)
    
    return Y_pred

def classification_metrics(Y_pred, Y_true):
	#TODO: Calculate the above mentioned metrics
	#NOTE: It is important to provide the output in the same order
   accuracy = accuracy_score(Y_true, Y_pred)
   auc = roc_auc_score(Y_true, Y_pred)
   precision = precision_score(Y_true, Y_pred)
   recall = recall_score(Y_true, Y_pred)
   f1score = f1_score(Y_true, Y_pred)
   return accuracy, auc, precision, recall, f1score

def get_acc_auc_kfold(X,Y,k=5):
    
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
    kf = KFold(X.shape[0], n_folds = k, random_state = RANDOM_STATE)
    #cross_validation module is deprecated
    sum_acc = 0
    sum_auc = 0
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_pred = my_classifier_predictions(X_train, Y_train, X_test)
        acc, auc, _, _, _ = classification_metrics(Y_pred,Y_test)
        sum_acc += acc
        sum_auc += auc
    return float(sum_acc)/k, float(sum_auc)/k


def main():
    X_train, Y_train, X_test = my_features()
    acc_k,auc_k = get_acc_auc_kfold(X_train,Y_train)
    print(auc_k)
    Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
    utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

	