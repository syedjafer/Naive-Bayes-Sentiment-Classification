from typing import Union
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel


app = FastAPI()
vector = load('vectors.joblib')
model = load('classifier.joblib')

class get_review(BaseModel):
    review: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/prediction")
def get_prediction(gr: get_review):
    text = [gr.review]
    vec = vector.transform(text)
    prediction = model.predict(vec)
    prediction = int(prediction)
    print(prediction)
    if prediction == 1:
        prediction = 'positive'
    else:
        prediction = 'negative'
    return {"sentence": gr.review, "prediction": prediction}
