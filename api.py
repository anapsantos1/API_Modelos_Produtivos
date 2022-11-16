from fastapi import FastAPI
from pydantic import BaseModel
import requests 
import mlflow 


# uvicorn api:app --reload
class Pinguins(BaseModel):
    island: int
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: int

app = FastAPI()

@app.get("/status")
def status():
    return {"status":"on"}

@app.get("/experiments")
def get_experiments():
    url = 'http://localhost:5001/api/2.0/preview/mlflow/experiments/list'
    response = requests.request('GET', url=url)
    dados = response.json()
    print(dados)
    return dados

@app.post("/model")
def predict(pinguins: Pinguins):
    mlflow.set_tracking_uri(uri='http://127.0.0.1:5001')
    PATH = 'models:/penguins/Production'
    classes = ['Adelie', 'Chinstrap', 'Gentoo']
    loaded_model = mlflow.sklearn.load_model(PATH)

    dados = [[p[1] for p in pinguins]]
    label = loaded_model.predict(dados) #list array [[],[],[]]

    resultado = classes[int(label[0])]
    return {'class': resultado}