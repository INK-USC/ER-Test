import pandas as pd
import os

def check_label_marking(df):
    df.loc[df['Answer.sentiment.label'] == "Positive", 'Answer.sentiment.label'] = "pos"
    df.loc[df['Answer.sentiment.label'] == "Negative", 'Answer.sentiment.label'] = "neg"
    df.loc[df['Answer.sentiment.label'] != df['Input.label'], 'Reject'] = "Incorrect Label Marking, kindly read the instructions before proceeding."

    return df

def check_rationale_marking(df):
    rationale_cols = [col for col in df.columns if "rationale_selection" in col]
    df['is_rationale_na'] = df[rationale_cols].isnull().apply(lambda x: all(x), axis=1)
    df.loc[df['is_rationale_na'] == True, 'Reject'] = "Word highlighting not done, kindly read the instructions before proceeding."

    return df


PATH = "../../data/mturk"

task_type = 1
num_samples = 200

if task_type == 1:
    file_name = "t"+str(task_type)+"_"+str(num_samples)+".csv"
    file_path = os.path.join(PATH, file_name)
    df = pd.read_csv(file_path)
    # df = check_label_marking(df)
    df = check_rationale_marking(df)
    write_file_name = "t"+str(task_type)+"_"+str(num_samples)+"_edited.csv"
    df.to_csv(os.path.join(PATH, write_file_name))

elif task_type == 2:
    file_name = "t"+str(task_type)+"_"+str(num_samples)+".csv"
    file_path = os.path.join(PATH, file_name)
    df = pd.read_csv(file_path)
    df = check_rationale_marking(df)
    write_file_name = "t"+str(task_type)+"_"+str(num_samples)+"_edited.csv"
    df.to_csv(os.path.join(PATH, write_file_name))

if task_type == 3:
    file_name = "t"+str(task_type)+"_"+str(num_samples)+".csv"
    file_path = os.path.join(PATH, file_name)
    df = pd.read_csv(file_path)
    df = check_label_marking(df)
    write_file_name = "t"+str(task_type)+"_"+str(num_samples)+"_edited.csv"
    df.to_csv(os.path.join(PATH, write_file_name))


