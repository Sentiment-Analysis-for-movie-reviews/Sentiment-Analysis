import streamlit as st
import json
import requests

class_names = ["joy","sadness","surprise","disgust","anger", "fear", "trust", "anticipation"]

st.title("💯Sentiment analysis for movie reviews based on Bert")
input_text = st.text_input("Please enter a movie review that you want to analyze: ")
epoch = st.selectbox("Choose the epoch ", [i for i in range(1, 11)])
if st.button("Submit"):
    if input_text:
        # files = {"file": ("review.txt", review_text)}
        data = {"review": input_text}
        try:
            response = requests.post(f"http://0.0.0.0:8080/sentiment_classification/{epoch}", json=data)
            response.raise_for_status()
            prediction = response.json()
            st.write(f'Review text: {input_text}')
            st.write(f'Sentiment  : {class_names[prediction.get("output")]}')
        except requests.exceptions.HTTPError as err:
            st.error(f"HTTP error occurred: {err}")
            st.error(response.content)
        except Exception as err:
            st.error(f"Error occurred: {err}")
    else:
        st.warning("Please enter a movie review.")
