import streamlit as st

st.write("Professor Loïc has assigned a school project that involves using Deep Learning to undertake, \
        one of \
        we choose the task **\"sentiment analysis\"**. We found to a dataset of movie reviews and their associated sentiments (positive, negative, or \
        neutral) and then preprocess the data to feed it into the Bert model. The Bert model \
        will be fine-tuned on the dataset and evaluated using various metrics such as \
        accuracy, precision, recall, and F1-score. Finally, the model's performance \
        will be visualized through confusion matrices and ROC curves. The project \
        aims to give students an understanding of how Deep Learning models can \
        be used for Natural Language Processing tasks such as sentiment analysis. \
        By the end of the project, students will have developed skills in data preprocessing, \
        model fine-tuning, and performance evaluation using the Bert model.")

st.write("l'objectif")

st.write("Les données utilisées (origine, format, statut juridique) et les traitements opérés sur celles-ci")



text = """
the [BlackLivesMatter Tweet Dataset](https://www.kaggle.com/datasets/carlsonhoo/baselinedataset) .

Contact : [<liupan.lucine@gmail.com>](mailto:liupan.lucine@gmail.com), [<ireneobasogie97@gmail.com>](mailto:ireneobasogie97@gmail.com)



## Model Training

We implement our classification model with transformers [Hugging Face](https://huggingface.co/docs/transformers/index), we use the basic `BertForSequenceClassification` model. 

If you want to train the model, please prepare the dataset and run the following command: 

```shell
python model_training.py
```

## Deploy Trained Model

Docker version 23.0.1 Docker-compose version 2.6.1

We deploy our trained model with docker by using FastAPI on the backend and Streamlit on the frontend. 

`Dockerfile` in corresponding directory gives the step to pull and bulid running image. The last line in the `Dockerfile` gives the command to run the container. 

If you want to run the Web App, please prepare the model and data in the backend, we have't processed them as database, you can directly save them in the directory. 

In the Web_App start the program.

```
docker-compose up
```
"""

st.write(text)