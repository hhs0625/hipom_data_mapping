import pandas as pd
import re
import os

# Loop through group numbers from 1 to 5
for group_number in range(1, 6):
    
    # Path to the train_all file
    train_all_path = f'data_preprocess/dataset/{group_number}/train_all.csv'
    
    # Read the train_all data
    train_all_csv = pd.read_csv(train_all_path, low_memory=False)
    
    # Concatenate tag_description based on the combination of thing and property
    tag_description_concatenated = train_all_csv.groupby(['thing', 'property'])['tag_description'].apply(lambda x: ' '.join(x)).reset_index()
    
    # Concatenate tag_name based on the combination of thing and property
    tag_name_concatenated = train_all_csv.groupby(['thing', 'property'])['tag_name'].apply(lambda x: ' '.join(x)).reset_index()
    
    # Calculate mapping_count
    mapping_count = train_all_csv.groupby(['thing', 'property']).size().reset_index(name='mapping_count')
    
    # Merge the three DataFrames: mapping_count, tag_description_concatenated, and tag_name_concatenated
    thing_property_grouped = pd.merge(mapping_count, tag_description_concatenated, on=['thing', 'property'])
    thing_property_grouped = pd.merge(thing_property_grouped, tag_name_concatenated, on=['thing', 'property'])
    
    # Calculate token_count by splitting tag_description using r'\S+'
    thing_property_grouped['td_token_count'] = thing_property_grouped['tag_description'].apply(lambda x: len(re.findall(r'\S+', x)))
    
    # Create pattern by replacing digits in 'thing' and 'property' with '#'
    thing_property_grouped['pattern'] = thing_property_grouped['thing'].str.replace(r'\d', '#', regex=True) + " " + thing_property_grouped['property'].str.replace(r'\d', '#', regex=True) 
    
    # Calculate the total number of unique thing_property combinations
    total_thing_property_count = thing_property_grouped.shape[0]
    
    # Specify the output path
    output_path = f'post_process/tfidf_class/0.class_document/{group_number}/sdl_class_rdoc.csv'
    
    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the result to the CSV file
    thing_property_grouped.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"Concatenated data saved to {output_path}")
    print(f"Total number of unique thing_property combinations: {total_thing_property_count}")
