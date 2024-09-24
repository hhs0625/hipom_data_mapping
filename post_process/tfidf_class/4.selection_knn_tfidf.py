import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import os
import numpy as np

k_accuracies = []

p_thing_str = 'c_thing'
p_property_str = 'c_property'

for k in range(5, 6):
    recall_list = []
    for group_number in range(1, 6):    
        test_csv = pd.read_csv(f'translation/0.result/{group_number}/test_p.csv', low_memory=False)
        test_csv = pd.read_csv(f'post_process/tfidf_class/0.class_document/distilbert/{group_number}/test_p_c_r.csv', low_memory=False)
        train_all_csv = pd.read_csv(f'data_preprocess/dataset/{group_number}/train_all.csv', low_memory=False)

        test_csv['s_score'], test_csv['s_thing'], test_csv['s_property'], test_csv['s_correct'] = -1, '', '', False
        duplicate_filtered = test_csv[test_csv['p_MDM']].copy()
        train_all_csv['tag_description'] = train_all_csv['tag_description'].fillna('')
        duplicate_filtered['tag_description'] = duplicate_filtered['tag_description'].fillna('')

        for ships_idx, group in duplicate_filtered.groupby('ships_idx'):
            for (p_thing, p_property), sub_group in group.groupby([p_thing_str, p_property_str]):
                matching_train_data = train_all_csv[(train_all_csv['thing'] == p_thing) & (train_all_csv['property'] == p_property)]
                if not matching_train_data.empty:
                    combined_descriptions = sub_group['tag_description'].tolist() + matching_train_data['tag_description'].tolist()

                    vectorizer = TfidfVectorizer(use_idf=True, token_pattern=r'\S+')
                    tfidf_matrix = vectorizer.fit_transform(combined_descriptions)

                    test_tfidf_matrix = tfidf_matrix[:len(sub_group)]
                    train_tfidf_matrix = tfidf_matrix[len(sub_group):]

                    distance_matrix = pairwise_distances(test_tfidf_matrix, train_tfidf_matrix, metric='cosine')
                    similarity_matrix = 1 - distance_matrix

                    for i, row in enumerate(similarity_matrix):
                        top_k_indices = np.argsort(row)[-k:]
                        sub_group.iloc[i, sub_group.columns.get_loc('s_score')] = row[top_k_indices].mean()
                else:
                    sub_group['s_score'] = 0

                duplicate_filtered.loc[sub_group.index, 's_score'] = sub_group['s_score']

        for ships_idx, group in duplicate_filtered.groupby('ships_idx'):
            for (p_thing, p_property), sub_group in group.groupby([p_thing_str, p_property_str]):
                best_index = sub_group.index.min() if (sub_group['s_score'] == -1).any() else sub_group['s_score'].idxmax()
                duplicate_filtered.at[best_index, 's_thing'] = sub_group.at[best_index, p_thing_str]
                duplicate_filtered.at[best_index, 's_property'] = sub_group.at[best_index, p_property_str]
                duplicate_filtered = duplicate_filtered.drop(sub_group.index.difference([best_index]))

        test_csv.update(duplicate_filtered[['s_thing', 's_property', 's_score']])
        test_csv['s_correct'] = ((test_csv['thing'] == test_csv['s_thing']) & 
                                 (test_csv['property'] == test_csv['s_property']) & 
                                 (test_csv['MDM']))

        mdm_true_count = test_csv['MDM'].sum()
        s_correct_count = test_csv['s_correct'].sum()
        recall = s_correct_count / mdm_true_count * 100
        recall_list.append(recall)

        if k == 5:
            output_path = f'post_process/0.result/{group_number}/test_s.csv'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            test_csv.to_csv(output_path, index=False)
            print(f"test_s.csv saved for Group {group_number} at {output_path}, mdm:{mdm_true_count}, correct:{s_correct_count}, recall:{recall:.2f}%")

    average_recall = np.mean(recall_list)
    k_accuracies.append(average_recall)
    print(f"k={k}, Average s_correct percentage: {average_recall:.2f}%")

overall_average_accuracy = np.mean(k_accuracies)
print(f"Overall average s_correct percentage across all k values: {overall_average_accuracy:.2f}%")

