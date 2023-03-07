Sentiment-Analysis
===================

Our project aims to analyze people's emotions on Black lives social movement. We use the [BlackLivesMatter Tweet Dataset](https://www.kaggle.com/datasets/carlsonhoo/baselinedataset) .

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

