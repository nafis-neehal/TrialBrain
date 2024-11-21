# from modules import data_loader

# data_loader.run_data_download()
# data_loader.push_to_hf_hub()

from modules.baseline_feature_loader import BaselineFeatureLoader

loader = BaselineFeatureLoader()
loader.run_data_download()
