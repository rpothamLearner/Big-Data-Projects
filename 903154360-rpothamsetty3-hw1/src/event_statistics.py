import time
import pandas as pd
import numpy as np
from datetime import timedelta
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    alive_pt = ~events['patient_id'].isin(mortality['patient_id'])
    events_alive = events[alive_pt]
    dead_pt = events['patient_id'].isin(mortality['patient_id'])
    events_dead = events[dead_pt]
    dead_pt_event_count = events_dead[['patient_id', 'event_id']].groupby(['patient_id'], as_index = False).count()
    avg_dead_event_count = dead_pt_event_count['event_id'].sum()/float(dead_pt_event_count.shape[0])
    max_dead_event_count = dead_pt_event_count['event_id'].max()
    min_dead_event_count = dead_pt_event_count['event_id'].min()
    alive_pt_event_count = events_alive[['patient_id', 'event_id']].groupby(['patient_id'], as_index = False).count()
    avg_alive_event_count = alive_pt_event_count['event_id'].sum()/float(alive_pt_event_count.shape[0])
    max_alive_event_count = alive_pt_event_count['event_id'].max()
    min_alive_event_count = alive_pt_event_count['event_id'].min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    alive_pt = ~events['patient_id'].isin(mortality['patient_id'])
    events_alive = events[alive_pt]
    dead_pt = events['patient_id'].isin(mortality['patient_id'])
    events_dead = events[dead_pt]
    dead_pt_encounter_count = events_dead[['patient_id', 'timestamp']].groupby(['patient_id'], as_index = False).agg({'timestamp':'nunique'})
    avg_dead_encounter_count = dead_pt_encounter_count['timestamp'].sum()/float(dead_pt_encounter_count.shape[0])
    max_dead_encounter_count = dead_pt_encounter_count['timestamp'].max()
    min_dead_encounter_count = dead_pt_encounter_count['timestamp'].min() 
    alive_pt_event_count = events_alive[['patient_id', 'timestamp']].groupby(['patient_id'], as_index = False).agg({'timestamp':'nunique'})
    avg_alive_encounter_count = alive_pt_event_count['timestamp'].sum()/float(alive_pt_event_count.shape[0])
    max_alive_encounter_count = alive_pt_event_count['timestamp'].max()
    min_alive_encounter_count = alive_pt_event_count['timestamp'].min() 

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    events.timestamp = pd.to_datetime(events.timestamp, format = '%Y-%m-%d')
    events = events.sort_values(by = ['patient_id','timestamp'])
    events = events[['patient_id', 'timestamp']].groupby(['patient_id'], as_index = False).agg({'timestamp': ['min', 'max']})
    events.columns = ['_'.join(col).strip() for col in events.columns.values]
    events['duration'] = events.timestamp_max - events.timestamp_min
    events = events.rename(columns = {'patient_id_' : 'patient_id'})
 
    alive_pt = ~events['patient_id'].isin(mortality['patient_id'])
    events_alive = events[alive_pt]
    dead_pt = events['patient_id'].isin(mortality['patient_id'])
    events_dead = events[dead_pt]
    seconds = timedelta(days=1).total_seconds()
    avg_dead_rec_len = events_dead.duration.mean().total_seconds()/seconds
    max_dead_rec_len = events_dead.duration.max().total_seconds()/seconds
    min_dead_rec_len = events_dead.duration.min().total_seconds()/seconds
    avg_alive_rec_len = events_alive.duration.mean().total_seconds()/seconds
    max_alive_rec_len = events_alive.duration.max().total_seconds()/seconds
    min_alive_rec_len = events_alive.duration.min().total_seconds()/seconds

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    You may change the train_path variable to point to your train data directory.
    OTHER THAN THAT, DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following line to point the train_path variable to your train data directory
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute event count metrics: " + str(end_time - start_time) + "s")
    print event_count

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute encounter count metrics: " + str(end_time - start_time) + "s")
    print encounter_count

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute record length metrics: " + str(end_time - start_time) + "s")
    print record_length
    
if __name__ == "__main__":
    main()
