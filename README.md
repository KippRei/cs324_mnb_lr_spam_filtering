# CS324-Comparison of Spam Analysis Using Multinomial Naive Bayes and Logistic Regression Models
Datasets:<br/>
spam_hard_ham_emails_dataset.csv: https://spamassassin.apache.org/old/publiccorpus/<br/>
spam_ham_emails_dataset.csv: https://www.kaggle.com/datasets/venky73/spam-mails-dataset?resource=download<br/><br/>

The main script is lr_nb_compare.py<br/>
csv_gen.py is used for generating the csv file from raw data (spam assassin just gives us raw data in multiple files)<br/><br/>

csv format example:<br/>
v1, v2<br/>
spam, "this is a spam email",<br/>
ham, "this is a ham email,...<br/><br/>

Usage:<br/>
1) Change DATASET_FP = '.\\data\\archive\\spam_ham_emails_dataset.csv' to point to spam-ham csv file location<br/>
2) All adjustable script parameters are capitalized at the top<br/>
3) When script finishes results will be printed to console<br/>
