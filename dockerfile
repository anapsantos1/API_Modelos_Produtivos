FROM python:3

RUN pip install mlflow-skinny
RUN pip install fastapi
RUN pip install requests
RUN pip install uvicorn


COPY . .

EXPOSE 80

CMD [ "uvicorn" , "api:app", "--host", "0.0.0.0", "--port", "80"]
