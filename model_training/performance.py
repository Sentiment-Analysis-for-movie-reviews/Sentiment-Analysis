import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer
# from .model_training import *
from model_training import get_df, encode_data, Data_loaders, BERT_Pretrained_Model, evaluate, tokenizer


# Defining our Performance Metrics: this is several methods in ClassificationEvaluation
class_names = ["joy","sadness","surprise","disgust","anger", "fear", "trust", "anticipation"]

def get_data(filepath: str, model_path: str):
    # Load the dataset
    df, label_dict = get_df(filepath)
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

    # Encode the dataset
    dataset_train, dataset_val = encode_data(df, tokenizer)

    dataloader_train, dataloader_val = Data_loaders(dataset_train, dataset_val)

    model = BERT_Pretrained_Model(label_dict)
    # If the cuda is available, we use cuda, otherwise we use 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device)))

    # We want to know if our model is overtraining
    val_loss , predictions, true_vals = evaluate(dataloader_val, model=model)

    # Flatten the data, cuz the predictions are composed of float numbers
    # which can not be compared with y_trues which vary from 0 to 8
    y_preds = np.argmax(predictions, axis=1).flatten()
    y_trues = true_vals.flatten()
    return y_preds, y_trues, model, label_dict

"""
We set up several performance metrics here, since the categories of movie review dataset are not balanced, so we previleged to use the method 'accuracy_per_class' to evaluate the model performance, while the categories of BlackLivesMatter and we previlege to get a classification report and confusion matrix
"""
class ClassificationEvaluation:
    def __init__(self, y_preds, y_trues):
        self.y_preds = y_preds
        self.y_trues = y_trues

    def f1_score_func(self):
        return f1_score(self.y_trues, self.y_preds, average='weighted')

    def accuracy_per_class(self, preds, labels, label_dict):
        # because the dataset we used is not balanced, there are much more "happy", so we decided to get the accuracy for each label
        label_dict_inverse = {v: k for k, v in label_dict.items()}
        print("shape of predict: ", preds.shape)
        print("shape of labels: ", labels.shape)
        preds_flat = preds
        labels_flat = labels.flatten()
        
        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat==label]
            y_true = labels_flat[labels_flat==label]
            print(f'Class:{label_dict_inverse[label]}')
            print(f'Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)}\n')

    def get_classification_report_confusion_matrix(self):
        ''' get classification report & confusion matrix '''
        cm = confusion_matrix(self.y_preds,self.y_trues)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names, yticklabels=class_names,
            title='Confusion matrix',
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        plt.show()
        plt.savefig('eval/confusion_matrix.png')
        # eval

        print('Classification report is being sent into eval/classification-report.txt...')
        with open('eval/classification-report.txt', 'w') as f:
            f.write(f'{classification_report(self.y_trues, self.y_preds, target_names=class_names)}')


if __name__=="__main__":
    # Task 10: Loading and Evaluating our Model
    filepath = "data/Black_dataset.csv"
    model_path = "model/Best_eval.model"
    y_preds, y_trues, model, label_dict = get_data(filepath=filepath, model_path=model_path)

    classificationEvaluation = ClassificationEvaluation(y_preds, y_trues)
    # because the dataset we used is not balanced, there are much more "happy", so we decided to get the accuracy for each label
    classificationEvaluation.accuracy_per_class(preds = y_preds, labels = y_trues, label_dict=label_dict)
    classificationEvaluation.get_classification_report_confusion_matrix()
