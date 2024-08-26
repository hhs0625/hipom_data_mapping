import shutil

source_file = 'data_import/raw_data.csv'

destination_file = 'data_preprocess/preprocessed_data.csv'

shutil.copy(source_file, destination_file)

print(f"File copied from {source_file} to {destination_file}")
