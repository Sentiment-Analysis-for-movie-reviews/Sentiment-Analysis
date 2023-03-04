import uvicorn
from fastapi import FastAPI
import pandas as pd
from fastapi import File, UploadFile, Response
from fastapi.encoders import jsonable_encoder
import json

# import inference
import interface


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.post("/visualization")
def label_visualization(file: UploadFile = File(...)):
    dataframe = pd.read_csv(file.file, delimiter = ',')
    # print("dataframe information: ", dataframe.head(10))
    output_dict = interface.Analyze_df(dataframe)
    return {"output": output_dict}

@app.post("/sentiment_classification/{epoch}")
def get_sentiment(epoch: int, text:dict):
    if epoch < 11:
        # review_text = file.file.read().decode("utf-8")
        review = text["review"]
        # print(review)
        output = interface.inference(review_text=review, No=epoch)
        return {"output": output}




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)