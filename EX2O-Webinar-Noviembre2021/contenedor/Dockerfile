FROM public.ecr.aws/lambda/python:3.8

COPY handler.py .

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["handler.predecir"]