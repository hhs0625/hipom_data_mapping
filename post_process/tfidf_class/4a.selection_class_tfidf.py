import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import os
import re
import numpy as np
import scipy.sparse as sp  # 추가된 부분

total_s_correct_count = 0
total_mdm_true_count = 0

# Modified TF-IDF Vectorizer to modify IDF behavior
class ModifiedTfidfVectorizer(TfidfVectorizer):
    def _tfidf_transform(self, X, copy=True):
        """Apply TF-IDF weighting to a sparse matrix X."""
        if not self.use_idf:
            return X
        df = np.bincount(X.indices, minlength=X.shape[1])
        n_samples, n_features = X.shape
        df += 1  # to smooth idf weights by adding 1 to document frequencies
        # Custom IDF: Logarithm of document frequency (df), rewarding common terms
        idf = np.log(df + 1)  # Modified IDF: log(1 + df)
        self._idf_diag = sp.diags(idf, offsets=0, shape=(n_features, n_features), format='csr')
        return X * self._idf_diag

for group_number in range(1, 6):

    test_path = f'translation/0.result/{group_number}/test_p.csv'
    ship_data_list_reference_doc_file_path = f'post_process/tfidf_class/0.class_document/{group_number}/sdl_class_rdoc.csv'

    test_csv = pd.read_csv(test_path, low_memory=False)
    sdl_rdoc = pd.read_csv(ship_data_list_reference_doc_file_path)

    test_csv['s_score'] = -1
    test_csv['s_thing'] = ''
    test_csv['s_property'] = ''
    test_csv['s_correct'] = False

    duplicate_filtered = test_csv[test_csv['p_MDM']].copy()

    thing_property_to_reference_doc = sdl_rdoc.set_index(['thing', 'property'])['tag_description'].to_dict()

    for ships_idx, group in tqdm(duplicate_filtered.groupby('ships_idx'), desc=f"Processing duplicates for group {group_number}"):
        for (p_thing, p_property), sub_group in group.groupby(['p_thing', 'p_property']):
            sub_group = sub_group.copy()
            tag_descriptions = sub_group['tag_description'].tolist()
            emtpy_ref = False
            reference_doc = thing_property_to_reference_doc.get((p_thing, p_property), '')
            if not reference_doc:
                p_pattern = sub_group['p_pattern'].iloc[0]
                sdl_match = sdl_rdoc[sdl_rdoc['pattern'] == p_pattern].sort_values(by='mapping_count', ascending=False).head(1)
                emtpy_ref = True
                if not sdl_match.empty:
                    reference_doc = sdl_match['tag_description'].iloc[0]
                else:
                    sub_group['s_score'] = 0
                    print(f"Reference document is empty for p_thing: {p_thing}, p_property: {p_property}")
                    duplicate_filtered.update(sub_group)
                    continue

            combined_descriptions = tag_descriptions + [reference_doc]

            vectorizer = ModifiedTfidfVectorizer(use_idf=True, token_pattern=r'\S+', ngram_range=(1, 1))
            tfidf_matrix = vectorizer.fit_transform(combined_descriptions)

            test_tfidf_matrix = tfidf_matrix[:-1]
            reference_vector = tfidf_matrix[-1]
            
            distance_matrix = pairwise_distances(test_tfidf_matrix, reference_vector.reshape(1, -1), metric='euclidean')  
            similarity_matrix = 1 - distance_matrix  

            sub_group['s_score'] = similarity_matrix.flatten()

            duplicate_filtered.loc[sub_group.index, 's_score'] = sub_group['s_score']

    for ships_idx, group in tqdm(duplicate_filtered.groupby('ships_idx'), desc=f"Processing duplicates for group {group_number}"):
        for (p_thing, p_property), sub_group in group.groupby(['p_thing', 'p_property']):
            if (sub_group['s_score'] == -1).any():
                best_index = sub_group.index.min()
            else:
                best_index = sub_group['s_score'].idxmax()
                row_position = sub_group.index.get_loc(best_index)

            duplicate_filtered.at[best_index, 's_thing'] = sub_group.at[best_index, 'p_thing']
            duplicate_filtered.at[best_index, 's_property'] = sub_group.at[best_index, 'p_property']

    test_csv.update(duplicate_filtered[['s_thing', 's_property', 's_score']])

    test_csv['s_correct'] = ((test_csv['thing'] == test_csv['s_thing']) & 
                             (test_csv['property'] == test_csv['s_property']) & 
                             (test_csv['MDM']))

    mdm_true_count = test_csv['MDM'].sum()
    s_correct_count = test_csv['s_correct'].sum()

    total_s_correct_count += s_correct_count
    total_mdm_true_count += mdm_true_count

    print(f"Group {group_number} - s_correct count: {s_correct_count}")
    print(f"Group {group_number} - MDM true count: {mdm_true_count}")
    print(f"Group {group_number} - s_correct percentage: {(s_correct_count / mdm_true_count) * 100:.2f}%")

    output_path = f'post_process/0.result/tfidf/{group_number}/test_s.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    test_csv.to_csv(output_path, index=False, encoding='utf-8-sig')

average_s_correct_percentage = (total_s_correct_count / total_mdm_true_count) * 100
print(f"Total s_correct count: {total_s_correct_count}")
print(f"Total MDM true count: {total_mdm_true_count}")
print(f"Average s_correct percentage across all groups: {average_s_correct_percentage:.2f}%")
