from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/predict")
def predict_form(
    pregnancies: int = Form(...),
    glucose: float = Form(...),
    bloodpressure: float = Form(...),
    skinthickness: float = Form(...),
    insulin: float = Form(...),
    bmi: float = Form(...),
    diabetespedigreefunction: float = Form(...),
    age: float = Form(...)
):
    data = [[pregnancies, glucose, bloodpressure, skinthickness,
             insulin, bmi, diabetespedigreefunction, age]]
    df = pd.DataFrame(data, columns=[
        "pregnancies", "glucose", "bloodpressure", "skinthickness",
        "insulin", "bmi", "diabetespedigreefunction", "age"
    ])
    prediction = model.predict(df)[0]

    message = (
        "⚠️ You may have diabetes. Please consult a doctor."
        if prediction == 1 else
        "✅ You are likely not diabetic."
    )

    return {"message": message}
