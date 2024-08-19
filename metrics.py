import joblib
import pandas as pdfrom sklearn.metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score


label_encoder = joblib.load("./emotional_dataset_label_encoder.enc"

def metric_pipeline(file_path):
    df = pd.read_csv(file_path)
    y_pred = label_encoder.transform(df['Y_pred'].to_list())
    y_val = label_encoder.transform(df['Y_val'].to_list())
    
    print("accuracy : {}\n".format(accuracy_score(y_val, y_pred)))
    print((classification_report(y_pred, y_val)))
    
    return df

def compare_label(df, label):
    full_answer = df['full_answer'].to_list()

    compare_label_list = list()
    for ans in full_answer:
        compare_label_list.extend([lb for lb in label if lb in ans])
    print("length of correct label",len(compare_label_list))
    print("length of original", len(full_answer))
    print("percentage : {}%".format(100*(len(compare_label_list)/len(full_answer))))
