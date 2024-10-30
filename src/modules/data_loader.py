import requests #for making API calls
import yaml #for reading the config file
import pandas as pd #for data manipulation
import logging #for logging
import os


# Load the parameters from the config file
with open('config.yml', 'r') as file:
    configs = yaml.safe_load(file)

# Set the logging configuration
def set_log_config(configs):
    '''
    Sets the logging configuration.
    '''
    logger = logging.getLogger(__name__)

    #check if log directory exists or create it
    data_loader_log_path = f"{configs['paths']['base_path']}/{configs['paths']['log_path']['data_loader']}"
    log_filename = "data_loader.log"
    if not os.path.exists(data_loader_log_path):
        os.makedirs(data_loader_log_path)

    logging.basicConfig(
        filename=f"{data_loader_log_path}/{log_filename}",
        filemode='a',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
        encoding='utf-8',
        level=logging.INFO,
        force=True
    )

set_log_config(configs)

# Define the functions to fetch data from the Clinical Trials API v2
def concat_fields_for_dataloader(fields):
    '''
    Concatenates the fields required for the data loader function.
    return: A string containing the fields to be fetched from the API.
    '''
    fields_string = '|'.join(fields)
    return fields_string

def fetch_api_v2_data(configs, **kwargs):
    '''
    Fetches all data from the Clinical Trials API v2.
    Filter used: the trial data has to have results (in order to have baseline measures)
    size: The number of maximum results/trials to return for each condition.
    return: A list of dictionaries (json format) containing the data for each condition.
    '''

    #this is the main URL of the API
    base_url = configs['data_loader']['base_url']

    #parameters of the API call
    ## pageSize - is the max number of trials fetched for each condition
    ## filter.advanced - requires all fetched trials to be - interventional, complete and have at least 6 or more baseline measures
    ## fields - is the fields/information of the trial we are fetching
    query_params = {
        'pageSize' : configs['data_loader']['max_size'],
        'filter.advanced': 'AREA[HasResults]true',
        'fields': concat_fields_for_dataloader(configs['data_loader']['fields'])
    }

    if 'nextPageToken' in kwargs:
        query_params['pageToken'] = kwargs['nextPageToken']

    #make the API call
    response = requests.get(base_url, params=query_params)

    if response.status_code == 200: #success
        # Process the response if successful and return JSON
        json_data = response.json()

    else:
        print(f"Error fetching data, status code: {response.status_code}")
        logging.error(f"Error fetching data, status code: {response.status_code}")

    return json_data

def fetch_api_v2_data_via_id_fields(nctid, configs):
    '''
    Fetches data from the Clinical Trials API v2 for a specific trial using the NCTID.
    nctid: The NCTID of the trial to search for.
    fields: The fields to be fetched from the API.
    return: A list of dictionaries (json format) containing the data for the trial.
    '''

    #this is the main URL of the API concatenated with the NCTID
    base_url = configs['data_loader']['base_url'] + f'/{nctid}'

    #parameters of the API call
    ## fields - is the fields/information of the trial we are fetching
    query_params = {
        'fields': concat_fields_for_dataloader(configs['data_loader']['fields'])
    }

    #make the API call
    response = requests.get(base_url, params=query_params)

    if response.status_code == 200: #success
        # Process the response if successful and return JSON
        json_data = response.json()
    else:
        print(f"Error fetching data, status code: {response.status_code}")

    return json_data

def parse_api_v2_data_to_df(data):

    '''
    Parses the data object returned from the Clinical Trials API v2, and converts it into a df.
    data: A list of dictionaries (json format) containing the data for each condition, returned by the API.
    return: A DataFrame containing the parsed data.
    '''

    # Create empty lists to store the extracted data
    nct_ids = []
    brief_titles = []
    eligibility_criteria = []
    brief_summaries = []
    conditions = []
    study_types = []
    interventions = []
    primary_outcomes = []
    measures = []
    overall_status = []

    # Iterate over the list of studies
    for study in data['studies']:

        #if any of the required fields are missing, skip the study, using try catch block
        try:

            # Extract the relevant data
            nct_ids.append(study['protocolSection']['identificationModule']['nctId'])
            brief_titles.append(study['protocolSection']['identificationModule']['briefTitle'])
            eligibility_criteria.append(study['protocolSection']['eligibilityModule']['eligibilityCriteria'])
            brief_summaries.append(study['protocolSection']['descriptionModule']['briefSummary'])
            overall_status.append(study['protocolSection']['statusModule']['overallStatus'])

            # Extract the conditions
            study_conditions = ''
            for condition in study['protocolSection']['conditionsModule']['conditions']:
                study_conditions += condition+ ', '
            conditions.append(study_conditions)

            # Extract the study type
            study_types.append(study['protocolSection']['designModule']['studyType'])

            # Extract the interventions
            study_interventions = []
            #check if the study has interventions
            if 'interventions' in study['protocolSection'].get('armsInterventionsModule', {}):
                for intervention in study['protocolSection']['armsInterventionsModule']['interventions']:
                    study_interventions.append(intervention['name'])
                interventions.append(', '.join(study_interventions))
            else:
                interventions.append('No Interventions')

            # Extract the primary outcomes
            study_primary_outcomes = []
            for outcome in study['protocolSection']['outcomesModule']['primaryOutcomes']:
                study_primary_outcomes.append(outcome['measure'])
            primary_outcomes.append(', '.join(study_primary_outcomes))

            # Extract the measures
            study_measures = []
            for measure in study['resultsSection']['baselineCharacteristicsModule']['measures']:
                study_measures.append(f"'{measure['title']}'")
            measures.append(', '.join(study_measures))

            # Create a DataFrame from the extracted data
            studies_df = pd.DataFrame({
                'NCTId': nct_ids,
                'BriefTitle': brief_titles,
                'EligibilityCriteria': eligibility_criteria,
                'BriefSummary': brief_summaries,
                'Conditions': conditions,
                'StudyType': study_types,
                'Interventions': interventions,
                'PrimaryOutcomes': primary_outcomes,
                'BaselineMeasures': measures,
                'OverallStatus': overall_status
            })

        except:
            #print(f"Error parsing data for study: {study['protocolSection']['identificationModule']['nctId']}")
            logging.error(f"Error parsing data for study: {study['protocolSection']['identificationModule']['nctId']}")
            continue

    return studies_df

def run_data_download():
    '''
    Runs the data scraping process.
    '''

    #fetch the first data batch (max 1000 trials)
    data = fetch_api_v2_data(configs)
    data_df = parse_api_v2_data_to_df(data)

    iter = 1 
    while 'nextPageToken' in data:
        data = fetch_api_v2_data(configs, nextPageToken=data['nextPageToken'])
        data_df = pd.concat([data_df, parse_api_v2_data_to_df(data)], ignore_index=True)
        print(f'Iteration: {iter}')
        iter += 1

    print(f"Total number of trials fetched: {data_df.shape[0]}")
    logging.info(f"Total number of trials fetched: {data_df.shape[0]}")

    save_dir = f"{configs['paths']['base_path']}/{configs['paths']['data_path']['raw']}"
    #check if the directory exists or create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = 'all_trials_withResults.csv'
    data_df.to_csv(f"{save_dir}/{file_name}", index=False)
