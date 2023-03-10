import streamlit as st
import plotly.express as px
import requests
import pandas as pd
from PIL import Image

st.subheader("This page shows the model performance and label distribution of our dataset")
if st.button("Performance of the best model"):
    image = Image.open('eval/confusion_matrix.png')
    st.image(image)
    with open('eval/classification-report.txt', 'r') as file:
        classification_report = file.read()
        st.caption("Here is the classification report:")
        st.text(classification_report)
        st.write("---------")

uploaded_file = st.file_uploader("Please upload **Black_dataset.csv** to show label distribution")
st.caption("Attention, **Black_dataset.csv** is the only file that can be processed")



data_visualization = "visualization"
if st.button("Submit"):
    if uploaded_file:
        files = {"file": uploaded_file.getvalue()}
        try:
            response = requests.post(f"http://0.0.0.0:8080/visualization", files=files)
            response.raise_for_status()
            dataframe_list = response.json().get("output")

            dataframe = pd.DataFrame(dataframe_list)
            st.write(dataframe.head(10))
            fig = px.histogram(dataframe, x="category")

            # Display the plot using Streamlit
            st.write("This graph shows category after cleaning")
            st.plotly_chart(fig)
            st.caption("This following information shows label distribution of uploaded dataset: ")
            st.write(dataframe.category.value_counts())
        except requests.exceptions.HTTPError as err:
            st.error(f"HTTP error occurred: {err}")
            st.error(response.content)
        except Exception as err:
            st.error(f"Error occurred: {err}")
    else:
        st.warning("Please enter a movie review.")

    

    