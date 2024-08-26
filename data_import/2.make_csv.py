import pandas as pd
import re

# Load the data_mapping CSV file
data_mapping_file_path = 'data_import/data_mapping.csv'  # Adjust this path to your actual file location
data_mapping = pd.read_csv(data_mapping_file_path, dtype=str)
df_master = pd.read_csv('data_import/data_model_master_export.csv')

# Generate patterns
data_mapping['thing_pattern'] = data_mapping['thing'].str.replace(r'\d', '#', regex=True)
data_mapping['property_pattern'] = data_mapping['property'].str.replace(r'\d', '#', regex=True)
data_mapping['pattern'] = data_mapping['thing_pattern'] + " " + data_mapping['property_pattern'] 
df_master['master_pattern'] = df_master['thing'] + " " + df_master['property']

# Create a set of unique patterns from master for fast lookup
master_patterns = set(df_master['master_pattern'])

# Check each pattern in data_mapping if it exists in df_master and assign the "MDM" field
data_mapping['MDM'] = data_mapping['pattern'].apply(lambda x: x in master_patterns)

# Remove specified fields
fields_to_remove = ['equip_type_code', 'tx_period', 'tx_type', 'on_change_yn', 'scaling_const', 'description', 'updated_time', 'status_code', 'is_timeout']
merged_data = data_mapping.drop(columns=fields_to_remove)

# Save the updated DataFrame to a new CSV file
output_file_path = 'data_import/raw_data.csv'
merged_data.to_csv(output_file_path, index=False, encoding='utf-8-sig')

print(f"Updated data saved to {output_file_path}")

# Filter the DataFrame where MDM is TRUE
data_mapping_mdm_true = merged_data[merged_data['MDM']]

# Save the filtered DataFrame to a new CSV file
mdm_true_output_file_path = 'data_import/data_mapping_mdm.csv'
data_mapping_mdm_true.to_csv(mdm_true_output_file_path, index=False, encoding='utf-8-sig')

print(f"MDM TRUE data saved to {mdm_true_output_file_path}")
