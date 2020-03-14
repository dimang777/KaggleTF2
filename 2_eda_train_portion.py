import pandas as pd

filename = '2_eda_train_portion.py'

load_folder = 'C:/Users/diman/Documents/Kaggle 2019 Data/tensorflow2/'

# shape: 50001 x 6
# columns: ['document_text', 'long_answer_candidates', 'question_text',
#           'annotations', 'document_url', 'example_id']

df_train_1 = pd.read_pickle(load_folder+'df_train_1.pkl')

df_train_1.iloc[0, :]
print(df_train_1.columns[0])
df_train_1.iloc[0, 0]
print(df_train_1.columns[1])
df_train_1.iloc[0, 1]
print(df_train_1.columns[2])
df_train_1.iloc[0, 2]
print(df_train_1.columns[3])
df_train_1.iloc[0, 3]
print(df_train_1.columns[4])
df_train_1.iloc[0, 4]
print(df_train_1.columns[5])
df_train_1.iloc[0, 5]

# Look at the nq_browser tool and eda in kaggle - that's sufficient for now
