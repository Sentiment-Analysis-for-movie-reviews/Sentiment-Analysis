# Loading modules
import torch
import pandas as pd
from tqdm import tqdm, trange


# import tensorflow as tf
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import AdamW, get_linear_schedule_with_warmup

import numpy as np
from sklearn.metrics import f1_score
import os
import random

import streamlit as st




epochs = 10
filepath = './../data/Black_dataset.csv'


def get_df(filepath: str) -> pd.DataFrame:
    # Task 1: Exploratory Data Analysis and Preprocessing
    if not os.path.exists(filepath):
        print("Path Error, Please make sure input right path of dataset")
        exit()

    # Process the training dataset
    df = pd.read_csv(filepath, delimiter=',')
    df = df.drop(0)
    df.insert(0, 'id', range(len(df)), allow_duplicates= False)
    df.rename(columns={'sentiment':'category', 'tweet_text': 'text'})
    # df = pd.read_csv(filepath,
    #     names=['id', 'category', 'text'])

    # Whether to modify the DataFrame rather than creating a new one.
    df.set_index('id', inplace=True)
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
    return df, label_dict


# See the distribution of each labels in training and validation set
# df.groupby(['category', 'label', 'data_type']).count()

# Task 3: Loading Tokenizer and Encoding our Data

def tokenizer():
    tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True
        )
    return tokenizer



def encode_data(df, tokenizer):
    encoded_data_train = tokenizer.batch_encode_plus(
        df[df.data_type=="train"].text.values,
        add_special_tokens=True, # pour que le modèle où une phrase commence et où elle termine
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors="pt",
        truncation=True
    )

    encoded_data_val = tokenizer.batch_encode_plus(
        df[df.data_type=="val"].text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors="pt",
        truncation=True
    )

    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train["attention_mask"] # a list of tensor "attention mask"
    labels_train = torch.tensor(df[df.data_type=="train"].label.values)

    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val["attention_mask"]
    labels_val = torch.tensor(df[df.data_type=="val"].label.values)


    dataset_train = TensorDataset(input_ids_train, attention_masks_train, 
                                    labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, 
                                labels_val)

    return dataset_train, dataset_val

# Task 5: Setting up BERT Pretrained Model
def BERT_Pretrained_Model(label_dict: dict):
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels = len(label_dict),
        output_attentions=False,# We don't want any unnecessary input from the model
        output_hidden_states=False # State just before the prediction, that might be useful for encoding 
    )

    return model


# Task 6: Creating Data Loaders

def Data_loaders(dataset_train, dataset_val):
    batch_size = 32 #32
    dataloader_train = DataLoader(
        dataset_train,
        sampler=RandomSampler(dataset_train),
        batch_size=batch_size
    )

    dataloader_val = DataLoader(
        dataset_val,
        sampler=SequentialSampler(dataset_val),
        batch_size=32 # We don't have many computation here, we don't do the backpropogation here
    )

    return dataloader_train, dataloader_val

# Task 7: Setting Up Optimizer and Scheduler
# optimizer is for setting the learning rate, "Adam Learning rate" is a way to optimize our learning rate
def Optimer_Scheduler_setup(model, dataloader_train):
    optimizer = AdamW(
        model.parameters(),
        lr=1e-5, # the original paper recommands 2e-5 > 5e-5
        eps=1e-8
    )

    epochs = 10

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader_train)*epochs # how many times we want our learning rate to change 
    )

    return optimizer, scheduler

# Task 8: Defining our Performance Metrics

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class:{label_dict_inverse[label]}')
        print(f'Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)}\n')


# Task 9: Creating our Training Loop
# seed_val = 17
# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# print(device)


# The 'evaluate' function does almost exactly what training does except we don't do backpropogation
# and that why we put it in evaluation mode (which freezes all our weights and you can also)
def evaluate(dataloader_val, model = None):


    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in tqdm(dataloader_val):

        batch = tuple(b.to(device) for b in batch)

        inputs = {"input_ids": batch[0],
                  "attention_mask": batch[1],
                  "labels": batch[2],
                 }
        # There is no gradient
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1] # We use logits as our predictions
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        # incase you are using a GPU and you want to pull a value onto your cpu in order to use them with numpy
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

def training_loop(dataloader_train,dataloader_val,optimizer,scheduler, model=None):
    best_F1 = 0
    for epoch in tqdm(range(1, epochs+1)):
        
        model.train() # set model to be training mode
        
        loss_train_total = 0
        
        progress_bar = tqdm(dataloader_train, 
                            desc="Epoch {:1d}".format(epoch),
                            leave=False,# let it overwrite itself after each new epoch
                            disable=False)
        # use batch to do backpropagation
        for batch in progress_bar:
            model.zero_grad()
            
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }
            
            outputs = model(**inputs) # unpack the dict straight into the model
            
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward() # Backpropogate
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # take the out gradient, give it a norm_value that we provide: 1
            # this prevents gradient
            
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

        if not os.path.exists('./../checkpoints/'):
            os.makedirs('./../checkpoints')

        torch.save(model.state_dict(), f'./../checkpoints/Bert_ft_epoch{epoch}.model')
        
        tqdm.write(f'\nEpoch {epoch}')
        
        loss_train_avg = loss_train_total/len(dataloader_train)
        tqdm.write(f'training loss: {loss_train_avg}')
        
        # We want to know if our model is overtraining
        val_loss , predictions, true_vals = evaluate(dataloader_val, model=model)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 score (weighted): {val_f1}')
        if val_f1 > best_F1:
            best_F1 = val_f1
            if not os.path.exists('./../model/'):
                os.makedirs('./../model')
            torch.save(model.state_dict(), f'./../model/Best_eval.model')


# Task 10: Loading and Evaluating our Model
def load_and_evaluate_our_model(dataloader_val):
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels = len(label_dict),
        output_attentions=False,# We don't want any unnecessary input from the model
        output_hidden_states=False # State just before the prediction, that might be useful for encoding 
    )

    model.to(device)

    model.load_state_dict(
        torch.load('./../model/Best_eval.model',
                map_location=torch.device('cuda')))

    _, predictions, true_vals = evaluate(dataloader_val, model= model)


    return accuracy_per_class(predictions, true_vals)



if __name__=="__main__":
    df, label_dict = get_df(filepath=filepath)
    tokenizer = tokenizer()
    dataset_train, dataset_val = encode_data(df, tokenizer)

    model = BERT_Pretrained_Model(label_dict)

    dataloader_train, dataloader_val = Data_loaders(dataset_train, dataset_val)

    optimizer, scheduler = Optimer_Scheduler_setup(model, dataloader_train)

    # Task 9: Creating our Training Loop
    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # training_loop(dataloader_train,dataloader_val,optimizer,scheduler, model=model)

    accuracy_per_class = load_and_evaluate_our_model(dataloader_val)
    print(accuracy_per_class)


    get_prediction = Get_prediction(review_text="Review text: # optimizer is for setting the learning rate, Adam Learning rate is a way to optimize our learning rate", tokenizer=tokenizer, device=device)
    get_prediction.get_tokenized_sentence(tokenizer)
    get_prediction.get_label()
    print(label_dict)


    
    
    
