import pandas as pd
import json

filename = '1_load.py'

load_folder = 'C:/Users/diman/Documents/Kaggle 2019 Data/tensorflow2/'
train_file = 'simplified-nq-train.jsonl'
test_file = 'simplified-nq-test.jsonl'

# Successful read in chunks
# df = pd.read_json(load_folder+test_file, lines=True)

# Because of memory issue, had to run up to three, stop, restart the computer,
# and run from the fourth. Just let the lines run through until reaching the
# new data samples to save (e.g., 4th df).
# Ran from 4th to 6th and put break there to stop. Restart. Finish 7th and 8th.

# Test - first ten
counter = 0
test = []
with open(load_folder+test_file, 'rt') as reader:
    oneline = json.loads(reader.readline())
    test.append(oneline)

    while counter<10:
        if not counter % 1:
            print(counter)
        test.append(oneline)
        counter += 1

    print('Finished!')
    reader.close()
print(type(test))
print(type(test[0]))
print(test[0])
    
    
df_train_1 = []
df_train_2 = []
df_train_3 = []
df_train_4 = []
df_train_5 = []
df_train_6 = []
df_train_7 = []
df_train_8 = []
counter = 1
with open(load_folder+test_file, 'rt') as reader:
    oneline = json.loads(reader.readline())
    df_train_1.append(oneline)

    while oneline:
        if not counter % 1:
            print(counter)
        if counter <= 50000:
            # df_train_1.append(oneline)
            if counter == 50000:
                print(counter)
            #     df_train_1 = pd.DataFrame(df_train_1)
            #     df_train_1.to_pickle(load_folder+'df_train_1.pkl')
        elif counter <= 100000:
            # df_train_2.append(oneline)
            if counter == 100000:
                print(counter)
            #     df_train_2 = pd.DataFrame(df_train_2)
            #     df_train_2.to_pickle(load_folder+'df_train_2.pkl')
        elif counter <= 150000:
            # df_train_3.append(oneline)
            if counter == 150000:
                print(counter)
            #     df_train_3 = pd.DataFrame(df_train_3)
            #     df_train_3.to_pickle(load_folder+'df_train_3.pkl')
            #     break
        elif counter <= 200000:
            # df_train_4.append(oneline)
            if counter == 200000:
                print(counter)
                # df_train_4 = pd.DataFrame(df_train_4)
                # df_train_4.to_pickle(load_folder+'df_train_4.pkl')
        elif counter <= 250000:
            # df_train_5.append(oneline)
            if counter == 250000:
                print(counter)
                # df_train_5 = pd.DataFrame(df_train_5)
                # df_train_5.to_pickle(load_folder+'df_train_5.pkl')
        elif counter <= 300000:
            # df_train_6.append(oneline)
            if counter == 300000:
                print(counter)
                # df_train_6 = pd.DataFrame(df_train_6)
                # df_train_6.to_pickle(load_folder+'df_train_6.pkl')
                # break
        elif counter <= 350000:
            df_train_7.append(oneline)

        try:
            oneline = json.loads(reader.readline())
        except json.JSONDecodeError:
            print('except')
            # if counter > 300000:
            #     df_train_7 = pd.DataFrame(df_train_7)
            #     df_train_7.to_pickle(load_folder+'df_train_7.pkl')
            #     oneline = []
        else:
            counter += 1

    print('Finished!')
    reader.close()
# The code and exceptions work. Checked
