import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

group_number = 1
# Load the CSV files
test_path = f'post_process/tfidf_class/0.class_document/{group_number}/test_p_c.csv'
test_path = f'post_process/tfidf_class/0.class_document/{group_number}/test_p_c_r.csv'
ship_data_list_reference_doc_file_path = f'post_process/tfidf_class/0.class_document/{group_number}/sdl_class_rdoc.csv'

test_csv = pd.read_csv(test_path, low_memory=False)
sdl_rdoc = pd.read_csv(ship_data_list_reference_doc_file_path)

# Initialize new columns in test_csv
test_csv['s_score'] = -1
test_csv['s_thing'] = ''
test_csv['s_property'] = ''
test_csv['s_correct'] = False

duplicate_filtered = test_csv[(test_csv['p_MDM'] == True)].copy()

# Create a mapping from thing/property to reference_doc
thing_property_to_reference_doc = sdl_rdoc.set_index(['thing', 'property'])['tag_description'].to_dict()

# Calculate s_score for duplicate rows
for ships_idx, group in tqdm(duplicate_filtered.groupby('ships_idx'), desc="Processing duplicates"):
    for (p_thing, p_property), sub_group in group.groupby(['p_thing', 'p_property']):
        sub_group = sub_group.copy()
        tag_descriptions = sub_group['tag_description'].tolist()
        
        # Get the reference document for the corresponding p_thing and p_property
        reference_doc = thing_property_to_reference_doc.get((p_thing, p_property), '')
        
        if reference_doc:
            # Combine the tag_descriptions and the reference_doc for fit_transform
            combined_descriptions = tag_descriptions + [reference_doc]
            
            # Create a new TF-IDF Vectorizer for this specific group
            vectorizer = TfidfVectorizer(
                token_pattern=r'\S+',
                norm='l2',  # Use L2 normalization
                ngram_range=(1, 7),  # Use both unigrams and bigrams
            )

            # Fit and transform the combined descriptions
            tfidf_matrix = vectorizer.fit_transform(combined_descriptions)
            
            # Separate the test_tfidf_matrix and reference_vector
            test_tfidf_matrix = tfidf_matrix[:-1]  # All but the last one
            reference_vector = tfidf_matrix[-1]    # The last one
            
            # Calculate the cosine similarity between the test descriptions and the reference_doc
            sub_group['s_score'] = cosine_similarity(test_tfidf_matrix, reference_vector).flatten()
        else:
            sub_group['s_score'] = 0
        
        # Update the s_score values back into the original test_csv
        duplicate_filtered.loc[sub_group.index, 's_score'] = sub_group['s_score']
   
for ships_idx, group in tqdm(duplicate_filtered.groupby('ships_idx'), desc="Processing duplicates"):
    for (p_thing, p_property), sub_group in group.groupby(['p_thing', 'p_property']):
        if (sub_group['s_score'] == -1).any():
            best_index = sub_group.index.min()
        else:
            # Find the index of the row with the highest s_score
            best_index = sub_group['s_score'].idxmax()
            row_position = sub_group.index.get_loc(best_index)

        # Assign s_thing and s_property only to the row with the highest s_score
        duplicate_filtered.at[best_index, 's_thing'] = sub_group.at[best_index, 'p_thing']
        duplicate_filtered.at[best_index, 's_property'] = sub_group.at[best_index, 'p_property']

# Now, update the original test_csv with the changes made in duplicate_filtered
test_csv.update(duplicate_filtered[['s_thing', 's_property', 's_score']])

# Calculate s_correct
test_csv['s_correct'] = ((test_csv['thing'] == test_csv['s_thing']) & 
                         (test_csv['property'] == test_csv['s_property']) & 
                         (test_csv['MDM']))

# Calculate the percentage of correct s_thing and s_property
mdm_true_count = test_csv['MDM'].sum()
s_correct_count = test_csv['s_correct'].sum()
s_correct_percentage = (s_correct_count / mdm_true_count) * 100

print(f"s_correct count: {s_correct_count}")
print(f"MDM true count: {mdm_true_count}")
print(f"s_correct percentage: {s_correct_percentage:.2f}%")


# Save the updated DataFrame to a new CSV file
output_path = test_path = f'post_process/0.result/{group_number}/test_s.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
test_csv.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"Updated data saved to {output_path}")

# Check for duplicates in s_thing and s_property within each ships_idx
print("\nShips_idx with duplicate s_thing and s_property:")
duplicate_ships_idx = []

for ships_idx, group in test_csv.groupby('ships_idx'):
    # Exclude rows with empty s_thing or s_property
    non_empty_group = group[(group['s_thing'] != '') & (group['s_property'] != '')]
    duplicate_entries = non_empty_group[non_empty_group.duplicated(subset=['s_thing', 's_property'], keep=False)]
    if not duplicate_entries.empty:
        duplicate_ships_idx.append(ships_idx)
        print(f"Ships_idx: {ships_idx}")
        print(duplicate_entries[['s_thing', 's_property']])

if not duplicate_ships_idx:
    print("No duplicates found.")
