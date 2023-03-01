import streamlit as st
import plotly.express as px
import requests
uploaded_file = st.file_uploader("Please update the file that you want to analyze")
st.caption("Attention, the acceptable formats are CSV, TSV and Excel")

data_visualization = 12
if st.button("Submit"):
    if uploaded_file:
        files = {"file": uploaded_file.getvalue()}
        try:
            dataframe = requests.post(f"http://backend:8080/{data_visualization}", files=files)
            dataframe.raise_for_status()
            dataframe = dataframe.json()
            st.write(dataframe)
            fig = px.histogram(dataframe, x="category")

            # Display the plot using Streamlit
            st.write("This graph shows category after cleaning")
            st.plotly_chart(fig)
            st.write(dataframe.category.value_counts())
        except requests.exceptions.HTTPError as err:
            st.error(f"HTTP error occurred: {err}")
            st.error(dataframe.content)
        except Exception as err:
            st.error(f"Error occurred: {err}")
    else:
        st.warning("Please enter a movie review.")

    

    