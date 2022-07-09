from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
from typing import Union

decision_tree = load("iris_classifier.joblib")

IRIS_CLASS_NAMES = {0: "Iris-Setosa", 1: "Iris-Versicolour", 2: "Iris-Virginica"}

app = FastAPI(__name__)

class Item(BaseModel):
    values: List[float]


@app.get("/")
def hello():
    return "Hello, World!"


@app.post("/score", methods=["POST"])
def score_inputs(content: Item):
    val_to_score = content.values

    result = decision_tree.predict([val_to_score])

    iris_name_result = IRIS_CLASS_NAMES[result[0]]

    return {"result": iris_name_result}
