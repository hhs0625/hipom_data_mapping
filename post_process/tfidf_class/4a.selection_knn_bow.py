import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
import os
import numpy as np

# Initialize overall accuracy results
k_accuracies = []

for k in range(1, 53):  # k를 1부터 52까지 수행
    total_s_correct_count = 0
    total_mdm_true_count = 0

    for group_number in range(1, 6):    
        # test_csv = pd.read_csv(f'post_process/tfidf_class/0.class_document/{group_number}/test_p_c.csv', low_memory=False)
        test_csv = pd.read_csv(f'translation/0.result/{group_number}/test_p.csv', low_memory=False)
        train_all_csv = pd.read_csv(f'data_preprocess/dataset/{group_number}/train_all.csv', low_memory=False)

        test_csv['s_score'], test_csv['s_thing'], test_csv['s_property'], test_csv['s_correct'] = -1, '', '', False
        duplicate_filtered = test_csv[test_csv['p_MDM']].copy()
        train_all_csv['tag_description'] = train_all_csv['tag_description'].fillna('')
        duplicate_filtered['tag_description'] = duplicate_filtered['tag_description'].fillna('')

        for ships_idx, group in duplicate_filtered.groupby('ships_idx'):
            for (p_thing, p_property), sub_group in group.groupby(['p_thing', 'p_property']):
                matching_train_data = train_all_csv[(train_all_csv['thing'] == p_thing) & (train_all_csv['property'] == p_property)]
                if not matching_train_data.empty:
                    combined_descriptions = sub_group['tag_description'].tolist() + matching_train_data['tag_description'].tolist()

                    # BoW 벡터화를 위한 CountVectorizer 사용
                    vectorizer = CountVectorizer(token_pattern=r'\S+')
                    bow_matrix = vectorizer.fit_transform(combined_descriptions).toarray()

                    test_bow_matrix = bow_matrix[:len(sub_group)]
                    train_bow_matrix = bow_matrix[len(sub_group):]

                    # 코사인 거리를 계산하고, 유사도로 변환 (1 - 거리)
                    distance_matrix = pairwise_distances(test_bow_matrix, train_bow_matrix, metric='euclidean')
                    similarity_matrix = 1 - distance_matrix

                    for i, row in enumerate(similarity_matrix):
                        top_k_indices = np.argsort(row)[-k:]  # 가장 가까운 k개의 인덱스 (유사도 기준, 내림차순)
                        sub_group.iloc[i, sub_group.columns.get_loc('s_score')] = row[top_k_indices].mean()  # 유사도를 s_score에 저장
                else:
                    sub_group['s_score'] = 0

                duplicate_filtered.loc[sub_group.index, 's_score'] = sub_group['s_score']

        for ships_idx, group in duplicate_filtered.groupby('ships_idx'):
            for (p_thing, p_property), sub_group in group.groupby(['p_thing', 'p_property']):
                best_index = sub_group.index.min() if (sub_group['s_score'] == -1).any() else sub_group['s_score'].idxmax()
                duplicate_filtered.at[best_index, 's_thing'] = sub_group.at[best_index, 'p_thing']
                duplicate_filtered.at[best_index, 's_property'] = sub_group.at[best_index, 'p_property']
                duplicate_filtered = duplicate_filtered.drop(sub_group.index.difference([best_index]))

        test_csv.update(duplicate_filtered[['s_thing', 's_property', 's_score']])
        test_csv['s_correct'] = ((test_csv['thing'] == test_csv['s_thing']) & 
                                 (test_csv['property'] == test_csv['s_property']) & 
                                 (test_csv['MDM']))

        mdm_true_count = test_csv['MDM'].sum()
        s_correct_count = test_csv['s_correct'].sum()

        total_s_correct_count += s_correct_count
        total_mdm_true_count += mdm_true_count

    if total_mdm_true_count > 0:
        average_s_correct_percentage = (total_s_correct_count / total_mdm_true_count) * 100
        k_accuracies.append(average_s_correct_percentage)
        print(f"k={k}, Average s_correct percentage: {average_s_correct_percentage:.2f}%")

# k의 평균 정확도 출력
overall_average_accuracy = np.mean(k_accuracies)
print(f"Overall average s_correct percentage across all k values: {overall_average_accuracy:.2f}%")
