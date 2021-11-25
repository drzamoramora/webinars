import pickle
import boto3
import io

# config para acceso a S3
s3client = boto3.client(
    's3',
    region_name='us-east-2'
)

# nombre del bucket de S3
bucketname = "ex2o-ml-models"

# abrir un recurso de S3
def open_s3(filename):

    s3 = boto3.resource('s3')
    return s3.Bucket(bucketname).Object(filename).get()['Body'].read()


# formato funcion compatible con Lambda
def predecir(event, context):

    # cargar el modelo de ML:
    modelo = pickle.loads(open_s3('modelo.pkl'))

    # cargar los parametros desde event en un tensor
    parametros = [[event['sepal_length'],event['sepal_width'],event['petal_length'],event['petal_width']]]

    # se ejecuta la preddiccion y se formatea como JSON obj
    prediccion = {'tipo de flor': int(modelo.predict(parametros)[0])}

    # retornamos el json
    return prediccion

# prueba local en su maquina
# python handler.py
# print(predecir({'sepal_length': 0.1, 'sepal_width' : 0.5, 'petal_length' : 1, 'petal_width' : 0.5}, None))