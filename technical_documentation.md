### Technical Documentation for Sentiment Analysis Project

Contact : [<liupan.lucine@gmail.com>](mailto:liupan.lucine@gmail.com), [<irene_obasogie@hotmail.com>](mailto:irene_obasogie@hotmail.com)

#### Introduction

The objective of this project is to perform Sentiment Analysis that aims to classify the sentiment of tweets about the ""Black Lives Matter"" movement using Deep Learning techniques. For this project, we used a dataset of tweets about [BlackLivesMatter movement](https://www.kaggle.com/datasets/carlsonhoo/baselinedataset), which was found from Kaggle.

The dataset was used to train a deep learning model using the **BERT** (Bidirectional Encoder Representations from Transformers) neural-network-based algorithm. BERT is a pre-trained language model developed by Google that has achieved state-of-the-art results in a variety of natural language processing tasks, including sentiment analysis. The BERT model is fine-tuned, using the labelled dataset of tweets about the "Black Lives Matter" movement to classify the sentiment of the text data. The project includes a back-end API built using **FastAPI** and a front-end application built using **Streamlit**.

#### Dataset

According to the website, the objective of the data collector was to conduct Emotion Analysis on the "Black Lives Matter"s social movement. The purpose is to understand how individuals feel about the topic and evaluate the emotions conveyed on Twitter. The chosen dataset enables fine-grained emotion analysis by utilizing Robert Plutchik's Wheel of Emotions, which includes "Fear", "Anger", "Sadness", "Joy", "Disgust", "Trust", "Surprise", and "Anticipation", rather than just sentiment polarity. Prior to utilizing th"Black Lives Matter"er dataset, a different small dataset containing movie reviews with imbalanced categories (with the majority of reviews being labelled as "happy") was utilized before. As expected, the results obtained from this dataset were unsatisfactory, and consequently, "Black Lives Matter"tter dataset was chosen as it represents an interesting subject matter and is already preprocessed. However, we didn't find the licence for the dataset.

The dataset is in CSV format and contains over 5,000 tweets collected from Twitter, being already preprocessed, which can be used directly to train the classification model. See this article [medium](https://carlson-hoo.medium.com/fine-grained-emotion-dataset-for-blacklivesmatter-70a21a4c5bdb) to find out how this dataset was created. The tweets were collected using the **GetOldTweet Twitter API** with keyword **#BlackLivesMatter**, and the dataset includes the following fields:

- **Sentiment**: The sentiment label assigned among joy, sadness, surprise, disgust, anger, fear, trust or anticipation.
- **Tweet_text**: Tweets with keyword **#BlackLivesMatter** being preprocessed (Exclude None English, Remove special characters, spelling correction, etc.)

#### Methodology

The methodology for this project involved several stages, including task selection, model selection, dataset selection, training, front-end and back-end implementations. Pan and Irene worked on this project and divided the work into several parts, Pan was in charge of model training and front-end implementation, Irene worked on dataset selection and back-end implementation.

Initially, we decided to use the BERT model for this project due to its effectiveness in natural language processing tasks. Two datasets were chosen for training the model, including a small movie review dataset and the "Black Lives Matter" dataset. However, after training the model on the movie review dataset, we realized that the categories were imbalanced, and the results were unsatisfactory. Consequently, we decided to switch to th"Black Lives Matter"er dataset, which was well-preprocessed and provided more balanced categories.

After training the model on the "Black Lives Matter" dataset, we obtained satisfactory results and utilized F1-score to save the best-performing model. We then implemented the front-end, using Streamlit with multiple pages to display the results of the emotion analysis. Additionally, we implemented the back-end, using FastAPI to handle requests from the front-end and execute the emotion analysis algorithm. How the front-end and back-end communicate with each other, was a time-consuming step for us when we were implementing the projet. We did some research and found that we need to use **requests**.

Lastly, we attempted to use Docker for containerization but found it unnecessary and that it would consume a huge memory, so we decided to drop it from the implementation. However, we still learned a lot from this practice. Throughout the project, we faced some problems and solved them by conducting research, consulting with one another, and making informed decisions based on our expertise and project goals. Overall, the methodology involved a collaborative and iterative approach, with each team member contributing their skills and knowledge to produce a successful outcome.

**Implementations**

The image below shows our code's structure:

<img title="" src="file:///Users/liupan/Library/Application%20Support/marktext/images/2023-03-07-18-42-23-image.png" alt="" width="221" data-align="center">

**Usage**
To use the Sentiment Analysis Project, follow these steps:

Installation:

1. Clone the project repository from GitHub.

```shell
git clone https://github.com/Sentiment-Analysis-for-movie-reviews/Sentiment-Analysis.git
```

2. Install the required dependencies using the pip package manager.

```shell
pip3 install -r requirements.txt
```

3. Run the back-end and Streamlit application using the command line interface.

**Data processing**

The downloaded file ```training-dataset.csv``` are in the ```./data``` folder, ```data_preparation.py``` process it and give the file ```Black_dataset.csv```, we use the processed dataset directly in our project.

**Model Training** 

We implemented our classification model with transformers [Hugging Face](https://huggingface.co/docs/transformers/index), we use the `BertForSequenceClassification` model. (You can skip this step, because the models are already trained.)

If you want to train the models by yourself, please prepare the dataset and run the following command:

```shell
python model_training.py
```

We evaluated our model with F1 score. The trained model in each epoch will be saved in ```./checkpoints``` and the most performing model will be saved in ```./model/Best_eval.model```

**Back-end API**
The back-end API of the Sentiment Analysis Project was built using FastAPI, a modern, fast (high-performant), web framework for building APIs with Python 3.7+ based on standard Python type hints. The API exposes endpoints for uploading and analyzing CSV files containing tweets about the "Black Lives Matter" movement. The API uses the pre-trained BERT model to classify the sentiment of the text data in the CSV files.

**Front-end Application**
The front-end application of the Sentiment Analysis Project was built using Streamlit, an open-source Python library that makes it easy to build interactive web applications for machine/deep learning and data science projects. The front-end application allows users to upload CSV file containing tweets about the "Black Lives Matter" movement and visualize the results of sentiment analysis performed by the back-end API. The front-end application provides an intuitive and user-friendly interface for interacting with the Sentiment Analysis Project.

We have designed a user interface (UI) that effectively showcases the functions of our model.

To start the back-end server, please run:

```shell
python ./Web_App/back-end/main.py
```

We start then the front-end, implemented by Streamlit. Please open a new terminal and run:

```shell
streamlit run ./Web_App/front-end/Hello.py
```

We can then get the link. Please click the link to test our model.

**Attention:** We trained for 10 epoches, so we obtained 10 models. Models consume a lot of space, and for that reason we decided to make available only two models for test: **Bert_ft_epoch1.model** and **Bert_ft_epoch10.model**. So when you test our models in the UI, please choose only 1 or 10 epoch.

For the page "Sentiment Analysis":

1. Open the Streamlit User Interface in a web browser.
2. Enter the text data to be analyzed and choose the epoch (1 or 10 only).
3. Click on the "Submit" button to initiate the Sentiment Analysis.
4. The results will be displayed.

For the page "Data Visualization":

1. Click "Performance of the best model" to see the confusion matrix and the classification report.

2. Upload ```Black_dataset.csv``` and click "Submit" to see the label distribution graph.

In the page "Technical Documentation", you will find the technical documentation of our project.

**Results**

We got an accuracy of 0.85. More evaluation metrics can be found in the classification report below.

<img title="" src="file:///Users/liupan/Library/Application%20Support/marktext/images/2023-03-07-18-28-27-image.png" alt="" width="383" data-align="center">

Confusion Matrix

<img title="" src="file:///Users/liupan/Desktop/Cours/M2_S2/réseau_de_neurones/project/new_dataset/Sentiment-Analysis/eval/confusion_matrix.png" alt="" width="439" data-align="center">

#### Conclusion

The Sentiment Analysis Project demonstrates the application of Neural Networks techniques for Sentiment Analysis of tweets about the "Black Lives Matter" movement. The project uses a preprocessed and labelled dataset of tweets and a Deep Learning model based on the BERT algorithm. The back-end API was built using FastAPI, and the front-end application was built using Streamlit, providing a complete and interactive solution for sentiment analysis of text data. 

**Improvements:** We can use the trained model to analyze other tweets or texts about "Black Lives Matter" movement. If we would like to go further with the project, we could try other pre-trained models to conduct sentiment analysis and compare their performance. If possible, we could implement Docker to contain our project.