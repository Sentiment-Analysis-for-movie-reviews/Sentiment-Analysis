import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome 👋")

st.sidebar.success("Select one function above.")

st.markdown(
    """
    Our project aims to analyze people's emotions on Black lives social movement. We use the [BlackLivesMatter Tweet Dataset](https://www.kaggle.com/datasets/carlsonhoo/baselinedataset).  
    **👈 Select a functionnality from the sidebar**  

    **Sentiment analysis**, also referred to as opinion mining, is an approach 
    to natural language processing (NLP) that identifies the emotional tone 
    behind a body of text. This is a popular way for organizations to determine 
    and categorize opinions about a product, service, or idea.
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.  

    

    #### Two functionnalities of our project
    - **Sentiment Analysis**: where you can enter a text/tweet about BlackLivesMatter movement, 
    and you'll get a sentiment prediction among 'joy', 'sadness', 'surprise', 'anger', 'fear', 'trust' and 'anticipation'
    - **Data Visualization**: where you can find the performance of the best model we got from our training loop and you can load the BlackLivesMatter dataset that we used for training our model and see the distribution graph of all sentiment labels.
    

    Contact: [<liupan.lucine@gmail.com>](mailto:liupan.lucine@gmail.com), [<ireneobasogie97@gmail.com>](mailto:ireneobasogie97@gmail.com)
    """
)