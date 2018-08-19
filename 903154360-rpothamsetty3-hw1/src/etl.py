import utils
import pandas as pd
import numpy as np
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    #deliverables_path = '/Users/rajeshpothamsetty/Dropbox/Rajesh_DB/IMP_Dropbox/OMSCS/homework1/deliverables'
    events.timestamp = pd.to_datetime(events.timestamp, format = '%Y-%m-%d')
    events = events.sort_values(by = ['patient_id','timestamp'])
    
    alive_pt = ~events['patient_id'].isin(mortality['patient_id'])
    events_alive = events[alive_pt]
    events_alive = events_alive.reset_index()

    events_alive = events_alive[['patient_id', 'timestamp']].groupby(['patient_id'], as_index = False).agg({'timestamp': 'max'})
    events_alive = events_alive.rename(columns = {'timestamp' : 'indx_date'})
    
    events_dead = mortality[['patient_id', 'timestamp']]
    events_dead = events_dead.rename(columns = {'timestamp' : 'indx_date'})
    events_dead['indx_date'] = pd.to_datetime(events_dead['indx_date']).apply(pd.DateOffset(-30))
    
    indx_date = pd.concat([events_alive, events_dead])
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    merge = pd.merge(events, indx_date, on = 'patient_id')
    merge['timestamp'] = pd.to_datetime(merge['timestamp'])
    merge['filter_date'] = pd.to_datetime(merge['indx_date']).apply(pd.DateOffset(-2000))
    merge['filter'] = pd.DataFrame([1 if merge['filter_date'][i] <= merge['timestamp'][i] <= merge['indx_date'][i] else 0 for i in range(len(merge)) ])
    
    filtered_events = merge[merge['filter'] == 1]
    filtered_events = filtered_events[['patient_id', 'event_id', 'value']]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    #feature_map_df = feature_map
    #filtered_events_df = filtered_events
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
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '/Users/rajeshpothamsetty/Dropbox/Rajesh_DB/IMP_Dropbox/OMSCS/BigData/HW1/homework1/deliverables/'
    deliverables_path = '../deliverables/'
    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    aggregated_events['value'] = list(zip(aggregated_events.feature_id, aggregated_events.feature_value))
    patient_features = aggregated_events.groupby('patient_id', as_index = False)['value'].agg({'value' : lambda value: list(value)})              
    patient_features = patient_features.set_index('patient_id')['value'].to_dict()
    mortality = mortality.set_index('patient_id')['label'].to_dict()

    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    #op_file = '../deliverables/features_svmlight.train'
    #op_deliverable = '../deliverables/features.train'
    
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    
    for key in patient_features:
        cur_line = ''
        if key in mortality.keys():
            cur_line = '1 '
        else:
            cur_line = '0 '
        #sorted_pt_features = sorted(patient_features[key], key = lambda x:x[0])
        patient_features[key].sort(key=lambda tup: tup[0])  # sorts in place
        for value in patient_features[key]:
            cur_line += str(int(value[0])) + ':' + str("{:.6f}".format(value[1])) + ' '
        deliverable1.write(cur_line)
        deliverable1.write('\n')
    
    for key in patient_features:
        cur_line = str(int(key)) + ' '
        if key in mortality.keys():
            cur_line += '1 '
        else:
            cur_line += '0 '
        patient_features[key].sort(key=lambda tup: tup[0])  # sorts in place
        for value in patient_features[key]:
            cur_line += str(int(value[0])) + ':' + str("{:.6f}".format(value[1])) + ' '
        deliverable2.write(cur_line)
        deliverable2.write('\n') 
    

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()