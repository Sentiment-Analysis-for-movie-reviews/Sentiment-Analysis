import uvicorn
from fastapi import FastAPI
import pandas as pd
from fastapi import File, UploadFile


# import inference
import interface


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.post("/visualization")
def label_visualization(file: UploadFile = File(...)):
    dataframe = pd.read_csv(file.file, names=["id", "category", "text"])
    output_dict = interface.Analyze_df(dataframe)
    return {"output": output_dict}

@app.post("/sentiment_classification/{epoch}")
def get_sentiment(epoch: int, text: dict):
    if epoch < 11:
        input_text = text["input_text"]
        prediction = interface.inference(input_text=input_text, No=epoch)
        return {"prediction": prediction}




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)