from typing import Dict, List
import pandas as pd
import logging
from .base_data_loader import BaseDataLoader


class BaselineFeatureLoader(BaseDataLoader):

    def __init__(self):
        """Initialize the baseline features data loader."""
        super().__init__()
        self.task_name = 'baseline_task'

    def get_required_fields(self) -> List[str]:
        """Return required fields for baseline features task."""
        return self.configs['baseline_data_loader']['fields']

    def _build_query_params(self, **kwargs) -> Dict:
        """Build query parameters specific to baseline features task."""
        params = {
            'pageSize': self.configs['baseline_data_loader']['max_size'],
            'filter.advanced': 'AREA[HasResults]true',
            'fields': self._concat_fields(self.get_required_fields())
        }

        if 'nextPageToken' in kwargs:
            params['pageToken'] = kwargs['nextPageToken']

        return params

    def parse_data_to_df(self, data: Dict) -> pd.DataFrame:
        """Parse API response for baseline features task."""
        parsed_data = {
            'NCTId': [],
            'BriefTitle': [],
            'EligibilityCriteria': [],
            'BriefSummary': [],
            'Conditions': [],
            'StudyType': [],
            'Interventions': [],
            'PrimaryOutcomes': [],
            'BaselineMeasures': [],
            'OverallStatus': []
        }

        for study in data.get('studies', []):
            try:
                protocol = study.get('protocolSection', {})
                nct_id = protocol.get('identificationModule', {}).get(
                    'nctId', 'Unknown')

                # Extract basic information
                parsed_data['NCTId'].append(nct_id)
                parsed_data['BriefTitle'].append(
                    protocol.get('identificationModule', {}).get('briefTitle', 'No Title'))
                parsed_data['EligibilityCriteria'].append(
                    protocol.get('eligibilityModule', {}).get('eligibilityCriteria', 'No Eligibility Criteria'))
                parsed_data['BriefSummary'].append(
                    protocol.get('descriptionModule', {}).get('briefSummary', 'No Summary'))
                parsed_data['OverallStatus'].append(
                    protocol.get('statusModule', {}).get('overallStatus', 'Unknown Status'))

                # Extract conditions
                conditions = protocol.get(
                    'conditionsModule', {}).get('conditions', [])
                parsed_data['Conditions'].append(
                    ', '.join(conditions) if conditions else 'No Conditions')

                # Extract study type
                parsed_data['StudyType'].append(
                    protocol.get('designModule', {}).get('studyType', 'Unknown Type'))

                # Extract interventions
                interventions = []
                if 'interventions' in protocol.get('armsInterventionsModule', {}):
                    interventions = [
                        intervention.get('name', '')
                        for intervention in protocol['armsInterventionsModule']['interventions']
                    ]
                parsed_data['Interventions'].append(
                    ', '.join(interventions) if interventions else 'No Interventions')

                # Extract primary outcomes
                outcomes = []
                if 'primaryOutcomes' in protocol.get('outcomesModule', {}):
                    outcomes = [
                        outcome.get('measure', '')
                        for outcome in protocol['outcomesModule']['primaryOutcomes']
                    ]
                parsed_data['PrimaryOutcomes'].append(
                    ', '.join(outcomes) if outcomes else 'No Primary Outcomes')

                # Extract baseline measures
                measures = []
                results_section = study.get('resultsSection', {})
                baseline_module = results_section.get(
                    'baselineCharacteristicsModule', {})

                if baseline_module and 'measures' in baseline_module:
                    measures = [
                        f"'{measure.get('title', '')}'"
                        for measure in baseline_module['measures']
                        if measure.get('title')
                    ]

                parsed_data['BaselineMeasures'].append(
                    ', '.join(measures) if measures else 'No Baseline Measures')

            except Exception as e:
                logging.error(
                    f"Error parsing data in {self.task_name} for study {nct_id}: {str(e)}")
                # If error occurs, append default values to maintain list lengths
                for key in parsed_data:
                    if len(parsed_data[key]) < len(parsed_data['NCTId']):
                        parsed_data[key].append(f"Error: {key} not available")

        # Verify all lists have the same length
        lengths = {key: len(value) for key, value in parsed_data.items()}
        if len(set(lengths.values())) > 1:
            logging.error(
                f"Inconsistent list lengths detected in {self.task_name}: {lengths}")
            raise ValueError(
                "Inconsistent data parsing resulted in lists of different lengths")

        # return after NULL removal and conversion to DataFrame
        df = pd.DataFrame(parsed_data)
        df.dropna(inplace=True)

        return df

    def _get_output_filename(self) -> str:
        """Return output filename for baseline features task."""
        # return 'all_trials_withResults.csv'
        return self.configs['baseline_data_loader']['output_filename']
