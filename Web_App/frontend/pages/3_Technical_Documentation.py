import streamlit as st
from PIL import Image



text1 = """
### Technical Documentation for Sentiment Analysis Project

Contact : [<liupan.lucine@gmail.com>](mailto:liupan.lucine@gmail.com), [<ireneobasogie97@gmail.com>](mailto:ireneobasogie97@gmail.com)

#### Introduction

The objective of this project is to perform Sentiment Analysist that aims to classify the sentiment of tweets about the Black Lives Matter movement using Deep Learning learning techniques. The project uses a dataset of tweets about [BlackLivesMatter mouvement](https://www.kaggle.com/datasets/carlsonhoo/baselinedataset), which was found from Kaggle.

The dataset was used to train a deep learning model using the **BERT** (Bidirectional Encoder Representations from Transformers) algorithm. BERT is a pre-trained language model developed by Google that has achieved state-of-the-art results in a variety of natural language processing tasks, including sentiment analysis. The BERT model is fine-tuned using the labeled dataset of tweets about the Black Lives Matter movement to classify the sentiment of the text data. The project includes a backend API built using **FastAPI** and a frontend application built using **Streamlit**.

#### Dataset

According to the website, the objective of data collector is to conduct Emotion Analysis on the Black Lives Matters social movement. The purpose is to undestand how individuals feel about the topic and evaluate the emotions conveyed on Twitter. The chosen dataset enables fine-grained emotion analysis by utilizing Robert Plutchik's Wheel of Emotions, which includes "Fear", "Anger", "Sadness", "Joy", "Disgust", "Trust", "Surprise", and "Anticipation", rather than just sentiment polarity. Prior to utilizing the Black Lives Matter dataset, a different small dataset containing movie reviews with imbalanced categories (with much more movie review tagged as "happy") was utilized before. However, the results obtained from this dataset were unsatisfactory, and consequently, the Black Lives Matter dataset was chosen as it represents an interesting subject matter and is already preprocessed. But we didn't find the licence for the dataset.

The dataset is in CSV format and contains over 5,000 tweets collected from Tweeter, beng already preprocessed, which can be used directly to train the classification model. See this article [medium](https://carlson-hoo.medium.com/fine-grained-emotion-dataset-for-blacklivesmatter-70a21a4c5bdb) to find out how this dataset created. The tweets were collected using the GetOldTweet Twitter API with keyword #BackLivesMatter, and the dataset includes the following fields:

- **Sentiment**: The sentiment label assigned among joy, sadness, surprise, disgust, anger, fear, trust or Anticipation.
- **Tweet_text**: Tweets with keyword #BackLivesMatter being preprocessed (Exclude None English, Remove special characters, spelling correction, etc.)

#### Methodology

The methodology for this project involved several stages, including task selection, model selection, dataset selection, training, and frontend and backend implementation. Pan and Irene worked on this project and divided the work into several parts, Pan was in charge of model training and frontend implementation, Irene worked on dataset selection and backend implementation.

Initially, we decided to use the BERT model for this project due to its effectiveness in natural language processing tasks. Two datasets were chosen for training the model, including a small movie review dataset and the Black Lives Matter dataset. However, after training the model on the movie review dataset, we realized that the categories were imbalanced, and the results were unsatisfactory. Consequently, we decided to switch to the Black Lives Matter dataset, which was well-preprocessed and provided more balanced categories.

After training the model on the Black Lives Matter dataset, we obtained satisfactory results and utilized F1-score to save the best-performing model. We then implemented a frontend using Streamlit with multiple pages to display the results of the emotion analysis. Additionally, we implemented a backend using FastAPI to handle requests from the frontend and execute the emotion analysis algorithm. How the frontend and backend communicate with each other is a time-consuming step for us when we implement the projet. We did some research and found that we need to use **requests**

Lastly, we attempted to use Docker for containerization but found it unnecessary and will take a huge memory when we need to pack it and send it to teacher, so we decided to drop it from the implementation. Howerver, we still learned a lot from this practice. Throughout the project, the we met some problems and solved them by conducting research, consulting with one another, and making informed decisions based on their expertise and project goals. Overall, the methodology involved a collaborative and iterative approach, with each team member contributing their skills and knowledge to produce a successful outcome.

**Implementations**

Our code structure can be found below: """

st.write(text1)


with open('code_structure.txt', 'r') as file:
    code_structure = file.read()
    st.text(code_structure)

text2 = """


**Usage**  

To use the Sentiment Analysis Project, follow these steps:

Installation:
To install the Sentiment Analysis Project, follow these steps:

1. Clone the project repository from GitHub.
2. Install the required dependencies using the pip package manager.
3. Run the Backend and Streamlit application using the command line interface.

**Data processing**

The downloaded file ```training-dataset.csv``` is in the ```./data``` folder, ```data_preparation.py``` process it and give the file ```Black_dataset.csv```, we use the processed dataset directly in our project.

install dependencies

```
pip3 install -r requirements.txt
```

**Model Training**

We implement our classification model with transformers [Hugging Face](https://huggingface.co/docs/transformers/index), we use the basic `BertForSequenceClassification` model.

If you want to train the model, please prepare the dataset and run the following command:

```shell
python model_training.py
```

We evaluated our model with F1 score, the trained model in each epoch will be saved in ```./checkpoints``` and the most performing model will be saved in ```./model/Best_eval.model```

**Backend API**  

The backend API of the Sentiment Analysis Project is built using FastAPI, a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints. The API exposes endpoints for uploading and analyzing CSV files containing tweets about the Black Lives Matter movement. The API uses the pre-trained BERT model to classify the sentiment of the text data in the CSV files.

**Frontend Application**  

The frontend application of the Sentiment Analysis Project is built using Streamlit, an open-source Python library that makes it easy to build interactive web applications for machine learning and data science. The frontend application allows users to upload CSV file containing tweets about the Black Lives Matter movement and visualize the results of sentiment analysis performed by the backend API. The frontend application provides an intuitive and user-friendly interface for interacting with the Sentiment Analysis Project.

We have designed a user interface (UI) that effectively showcases the functions of our model.

To start the backend server, please run:

```shell
python ./Web_App/backend/main.py
```

We start than the frontend implemented by streamlit, please open a new terminal and run:

```shell
streamlit run ./Web_App/frontend/Hello.py
```

We can than get the link, please click the link to test our model.

Attention: we trained for 10 epoches, so we obtained 10 models, for the reason that models takes a lot of space, we decided to send you only two models **Bert_ft_epoch1.model** and **Bert_ft_epoch10.model**, so when you test our models in the UI, please choose only 1 or 2 epoch.

For the page "Sentiment Analysis"

1. Open the Streamlit user interface in a web browser.
2. Enter the text data to be analyzed and choose the epoch
3. Click on the "Submit" button to initiate the sentiment analysis.
4. The results will be displayed.

For the page "Data Visualization"

1. Click "Performance of the best model" to see confusion matrix and classification report
  
2. Upload ```Black_dataset.csv``` and click "Submit" to see label distribution graph
  

For the page "Technical Documentation", you will find technical documentation of our project.

**Results**

We got an overall accuracy of 0.85"""
st.write(text2)

image = Image.open('eval/confusion_matrix.png')
st.image(image)
with open('eval/classification-report.txt', 'r') as file:
    classification_report = file.read()
    st.caption("Here is the classification report:")
    st.text(classification_report)
    st.write("---------")
text3 = """
#### Conclusion

The Sentiment Analysis Project is a school project that demonstrates the application of machine learning techniques for sentiment analysis of tweets about the Black Lives Matter movement. The project uses a preprocessed and labeled dataset of tweets and a machine learning model based on the BERT algorithm. The backend API is built using FastAPI, and the frontend application is built using Streamlit, providing a complete and interactive solution for sentiment analysis of text data.

Improvements: We can use the trained model to analyze other tweets or texts about BlackLivesMatter mouvement. If we would like to further the project, we can try other pre-trained model to conduct sentiment analysis and compare thier performance, if possible, we implement Docker to contain our project.
"""

st.write(text3)
