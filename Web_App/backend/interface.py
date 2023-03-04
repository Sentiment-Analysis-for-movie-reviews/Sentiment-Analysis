from transformers import BertForSequenceClassification, BertTokenizer
from Get_prediction import Get_prediction
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

def BERT_Pretrained_Model(label_dict: dict):
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels = len(label_dict),
        output_attentions=False,# We don't want any unnecessary input from the model
        output_hidden_states=False # State just before the prediction, that might be useful for encoding 
    )

    return model

def Analyze_df(df: pd.DataFrame) -> pd.DataFrame:
    # Task 1: Exploratory Data Analysis and Preprocessing
    # Whether to modify the DataFrame rather than creating a new one.
    df = df.drop(0)
    df.insert(0, 'id', range(len(df)), allow_duplicates= False)
    df = df.rename(columns={'sentiment':'category', 'tweet_text': 'text'})
    df.set_index('id', inplace=True)

    # df = df[~df.category.str.contains("\|")]
    # df = df[df.category != 'nocode']
    possible_labels = df.category.unique()
    label_dict = {}

    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    df["label"] = df.category.replace(label_dict)

    # Task 2: Training/Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        df.index.values,
        df.label.values,
        test_size=0.15,
        random_state=17,
        stratify=df.label.values # divide all categories with the set proportion
    )

    df['data_type'] = ['not_set']*df.shape[0]

    df.loc[X_train, 'data_type'] = 'train'
    df.loc[X_val, 'data_type'] = 'val'
    # Convert DataFrame to dictionary using the 'records' orientation
    output_dict = df.to_dict(orient='records')

    return output_dict

def inference(review_text, No):
    model_name = f"./checkpoints/Bert_ft_epoch{No}.model"
    label_dict = {0: 'joy', 1: 'sadness', 2: 'superise', 3: 'disgust', 4: 'anger', 5: 'fear', 6: 'trust', 7: 'anticipation'}
    model = BERT_Pretrained_Model(label_dict)
    device = torch.device('cpu')
    model.load_state_dict(torch.load(model_name, map_location=torch.device(device=device)))
    tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True
        )
    get_prediction = Get_prediction(review_text,tokenizer, device)
    prediction = get_prediction.get_label(model)
    # print("test the pridcition: ", prediction)
    return prediction
