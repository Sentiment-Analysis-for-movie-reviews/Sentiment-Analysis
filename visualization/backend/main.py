import uvicorn
from fastapi import FastAPI
import pandas as pd
from fastapi import File, UploadFile, Response
from fastapi.encoders import jsonable_encoder

# import inference
import interface

"""
This is our server. FastAPI creates two endpoints, one dummy ("/") and one for serving our prediction ("/{Prediction}"). The serving endpoint takes in a name as a URL parameter. We're using nine different trained models to perform style transfer, so the path parameter will tell us which model to choose. The image is accepted as a file over a POST request and sent to the inference function. Once the inference is complete, the file is stored on the local filesystem and the path is sent as a response.
"""

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}



@app.post("/{epoch}")
def get_sentiment(epoch: int, file: UploadFile = File(...)):
    if epoch < 11:
        review_text = file.file.read().decode("utf-8")
        output = interface.inference(review_text=review_text, No=epoch)
        return {"output": output}
    else:
        dataframe = pd.read_csv(file.file, names=['id', 'category', 'text'])
        # print("dataframe information: ", dataframe.head(10))
        output_dict = interface.Analyze_df(dataframe)
        return {"output": output_dict}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)