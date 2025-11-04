'''This script analyzes and compares the accuracy of different methods of detecting spam email.
The algorithms we are comparing are: Multinomial Naive Bayes (MNB) and Logistic Regression (LR).
We also look at the hierarchical approach by looking at how MNB->LR and LR->MNB performance.
PARAMETERS:
MIN_THRESHOLD/MAX_THRESHOLD = The range of confidence thresholds we want to test;
MIN_MAX_FEATURES/MAX_MAX_FEATURES = The range of max_features for the tf-idf matrix;
SAMPLE_SIZES = Sample size percentages for dataset (i.e. np.linspace(0.1, 1.0, 10) tests from 10% to 100% of dataset samples)'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

### SCRIPT PARAMETERS ###
# Confidence threshold
MIN_THRESHOLD = 0.2
MAX_THRESHOLD = 0.9
THRESHOLD_STEP = 0.1
# Max number of words considered for tf-idf
MIN_MAX_FEATURES = 100
MAX_MAX_FEATURES = 15000
MAX_FEATURES_STEP = 100
# Sample size proportions
SAMPLE_SIZES = np.linspace(0.1, 1.0, 10) # Testing from 10% to 100%
# Seed for reproducible results (set to None for randomized seed)
RANDOM_STATE = None

# To hold highest accuracy numbers (Accuracy %, Num of Samples, Confidence Threshold, Max TF-IDF Features)
max_mnb = (0, 0, 0, 0)
max_lr = (0, 0, 0, 0)
max_mnb_lr = (0, 0, 0, 0)
max_lr_mnb = (0, 0, 0, 0)

# Helper function for float range (Geeks for Geeks)
def float_range(start, stop, step):
    while start < stop:
        yield start
        start += step

# Load the dataset (with 'v1'/'v2' columns)
try:
    df = pd.read_csv('C:\\Users\\jappa\\Repos\\cs324\\data\\archive\\spam_ham_emails_dataset.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
except FileNotFoundError:
    print("Error: 'spam.csv' not found. Please check your file path.")
    exit()
except KeyError:
    print("Error: DataFrame columns not found. Ensure file has 'v1' and 'v2' columns.")
    exit()

X = df['text']
y = df['label']

# Create test set
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Calculate total number of tests for tqdm progress bar
total_num_max_feat_tests = len(range(MIN_MAX_FEATURES, MAX_MAX_FEATURES, MAX_FEATURES_STEP))
total_num_threshold_tests = len(list(float_range(MIN_THRESHOLD, MAX_THRESHOLD, THRESHOLD_STEP)))
total_num_samples_tests = len(SAMPLE_SIZES)
tqdm_total = total_num_max_feat_tests * total_num_threshold_tests * total_num_samples_tests

with tqdm(total=tqdm_total) as progress_bar:
    for MAX_FEATURES in range(MIN_MAX_FEATURES, MAX_MAX_FEATURES, MAX_FEATURES_STEP):
        # Initialize and Fit TF-IDF once for this MAX_FEATURES
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=MAX_FEATURES)
        tfidf_vectorizer.fit(X_train_full)

        # Pre-Transform the full training and test sets
        X_train_vectorized_full = tfidf_vectorizer.transform(X_train_full)
        X_test_vectorized = tfidf_vectorizer.transform(X_test)

        # Pre-calculate indices for quick sampling outside the inner loop
        # Create the full list of indices to sample from (0 to N-1)
        full_indices = np.arange(len(X_train_full))

        for THRESHOLD in float_range(MIN_THRESHOLD, MAX_THRESHOLD, THRESHOLD_STEP):
            # Define the sample sizes (proportions of the full training set) to test

            results = []

            for size_proportion in SAMPLE_SIZES:

                # Calculate the actual number of samples
                sample_size = int(size_proportion * len(X_train_full))

                # Randomly sample the training data
                if size_proportion == 1.0:
                    # Use the pre-vectorized full data
                    X_sample_vectorized = X_train_vectorized_full
                    y_sample = y_train_full
                else:
                    # Get the indices for the sample (fast operation)
                    sample_indices, _, _, _ = train_test_split(
                        full_indices, y_train_full, train_size=sample_size,
                        random_state=RANDOM_STATE, stratify=y_train_full
                    )
                    # Subset the pre-vectorized data (fast operation on sparse matrix)
                    X_sample_vectorized = X_train_vectorized_full[sample_indices]
                    y_sample = y_train_full.iloc[sample_indices]

                # --- 1. Train and Evaluate Base Models ---
                
                # MNB
                mnb_model = MultinomialNB()
                mnb_model.fit(X_sample_vectorized, y_sample)
                y_pred_mnb = mnb_model.predict(X_test_vectorized)
                y_proba_mnb = mnb_model.predict_proba(X_test_vectorized)[:, 1]
                accuracy_mnb = accuracy_score(y_test, y_pred_mnb)

                # LR
                lr_model = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE)
                lr_model.fit(X_sample_vectorized, y_sample)
                y_pred_lr = lr_model.predict(X_test_vectorized)
                y_proba_lr = lr_model.predict_proba(X_test_vectorized)[:, 1]
                accuracy_lr = accuracy_score(y_test, y_pred_lr)


                # --- 2. Hierarchical: LR Filter -> MNB Refinement ---

                # Identify indices where LR predicts spam with high confidence
                mnb_refine_indices = np.where(y_proba_lr > THRESHOLD)[0]

                # Start with LR predictions as base
                y_pred_hierarchical_lr_mnb = y_pred_lr.copy()
                
                if len(mnb_refine_indices) > 0:
                    # Refine prediction using MNB on the subset
                    X_test_mnb_subset = X_test_vectorized[mnb_refine_indices, :]
                    y_pred_mnb_subset = mnb_model.predict(X_test_mnb_subset)
                    y_pred_hierarchical_lr_mnb[mnb_refine_indices] = y_pred_mnb_subset

                accuracy_hierarchical_lr_mnb = accuracy_score(y_test, y_pred_hierarchical_lr_mnb)


                # --- 3. Hierarchical: MNB Filter -> LR Refinement ---

                # Identify indices where MNB predicts spam with high confidence
                lr_refine_indices = np.where(y_proba_mnb > THRESHOLD)[0]

                # Start with MNB predictions as base
                y_pred_hierarchical_mnb_lr = y_pred_mnb.copy()

                if len(lr_refine_indices) > 0:
                    # Refine prediction using LR on the subset
                    X_test_lr_subset = X_test_vectorized[lr_refine_indices, :]
                    y_pred_lr_subset = lr_model.predict(X_test_lr_subset)
                    y_pred_hierarchical_mnb_lr[lr_refine_indices] = y_pred_lr_subset

                accuracy_hierarchical_mnb_lr = accuracy_score(y_test, y_pred_hierarchical_mnb_lr)

                if accuracy_mnb > max_mnb[0]:
                    max_mnb= accuracy_mnb, sample_size, THRESHOLD, MAX_FEATURES
                if accuracy_lr > max_lr[0]:
                    max_lr = accuracy_lr, sample_size, THRESHOLD, MAX_FEATURES
                if accuracy_hierarchical_lr_mnb > max_lr_mnb[0]:
                    max_lr_mnb = accuracy_hierarchical_lr_mnb, sample_size, THRESHOLD, MAX_FEATURES
                if accuracy_hierarchical_mnb_lr > max_mnb_lr[0]:
                    max_mnb_lr = accuracy_hierarchical_mnb_lr, sample_size, THRESHOLD, MAX_FEATURES
                
                progress_bar.update(1)

                # # Uncomment for single test visualization
                # # --- 4. Store Results ---
                # results.append({
                #     'sample_proportion': size_proportion,
                #     'sample_n': sample_size,
                #     'MNB_Accuracy': accuracy_mnb,
                #     'LR_Accuracy': accuracy_lr,
                #     'Hierarchical_LR_MNB': accuracy_hierarchical_lr_mnb,
                #     'Hierarchical_MNB_LR': accuracy_hierarchical_mnb_lr
                # })
                # print("Comparison complete.")

                # results_df = pd.DataFrame(results)
                # print("\nAccuracy Comparison Table:")
                # print(results_df)

# Print max accuracy score results for each algo
print('Algorithm: (Accuracy %, Num of Samples, Confidence Threshold, Max TF-IDF Features)')
print(f'MNB: {max_mnb}')
print(f'LR: {max_lr}')
print(f'LR->MNB: {max_lr_mnb}')
print(f'MNB->LR: {max_mnb_lr}')

# # # --- 5. Visualization (uncomment if running single test to display graph) ---
# plt.figure(figsize=(12, 7))
# plt.plot(results_df['sample_n'], results_df['MNB_Accuracy'], marker='o', label='1. MNB (Baseline)')
# plt.plot(results_df['sample_n'], results_df['LR_Accuracy'], marker='x', label='2. LR (Baseline)')
# plt.plot(results_df['sample_n'], results_df['Hierarchical_LR_MNB'], marker='s', label='3. Hierarchical (LR Filter -> MNB)')
# plt.plot(results_df['sample_n'], results_df['Hierarchical_MNB_LR'], marker='^', label='4. Hierarchical (MNB Filter -> LR)')

# plt.title('Performance Comparison of Base and Hierarchical Classifiers vs. Sample Size')
# plt.xlabel('Training Sample Size (Number of Emails)')
# plt.ylabel('Accuracy on Fixed Test Set')
# plt.legend()
# plt.grid(True)
# plt.show()