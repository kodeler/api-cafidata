from distutils.log import error
import os
import firebase as fb
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from model import Model
from data import Data 


app = Flask(__name__)
app.config["DEBUG"] = True
# sirve para


CORS(app, resources={r"/*": {"origins": "*"}})# cors sirve para que se pueda acceder a la api desde cualquier lugar


#@app.after_request
#def after_request(response):
  #response.headers.add('Access-Control-Allow-Origin', '*')
  #response.headers.add("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept")
  #response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  #response.headers.add('Access-Control-Allow-Credentials', 'true')
  #return response

@app.route("/", methods=["GET"])# se crea una ruta para la api que se va a utilizar para obtener los datos de la api
def home():
    return """
        <!DOCTYPE html>
        <body style="width: 880px; margin: auto;">
        <h1> CAFIDATA API </h1>
        <p>Welcome to API for CAFIDATA </p>
        </body>
            """


@app.route("/predict", methods=["POST"])  # GET requests will be blocked
def predict_from_model():
    req_data = request.get_json()
    # Brackets require these fields to be present
    # Sort of a safety contract to ensure we always have valid path
    uid = req_data["uid"]  # user id from firebase auth system (unique) 
    pid = req_data["projId"]  # project id from firebase auth system (unique)
    model = req_data["model"]  # desired model name
    inputs = req_data["inputs"]  # df vars / x_predict

    model = Model.load_model(str(uid), str(pid), str(model))# se carga el modelo que se quiere utilizar para predecir los datos
    prediction = model.predict(inputs)# se hace la prediccion de los datos que se quieren predecir con el modelo
    return jsonify(prediction)# se retorna la prediccion de los datos en formato json para que se pueda utilizar en el frontend


# assume csv has been made, now upload models
@app.route("/store", methods=["POST"])# se crea una ruta para la api que se va a utilizar para almacenar los datos de la api
def store():

    
    # once we know what implement fetching the csv from firebase, that'll go in the call
    # nosotros solo necesitamos el csv que se ha creado en la carpeta data y que se ha subido a firebase 
    req_data = request.get_json()

    # Brackets require these fields to be present
    # Sort of a safety contract to ensure we always have valid path
    uid = req_data["uid"]  # user id
    proj_id = req_data["projId"]  # unique project hash
    model_list = req_data["modelList"]  # list of models user uses
    target_param = req_data["targetParameter"]  # output
    df_vars = req_data["dfVariables"]  # inputs
    csv_path = req_data["csvPath"]  # name of uploaded csv
    nan_method = req_data["nanMethod"]  # method for dealing with NaN's

    # Get firebase stuff
    bucket = fb.bucket_init()

    df = fb.get_csv(bucket, csv_path)# se obtiene el csv que se ha subido a firebase 

    data = Data(df, target_param, df_vars, nan_method)# se crea un objeto data con los datos que se han subido a firebase
    X_train, X_test, y_train, y_test = data.get_train_test_split()# se obtienen los datos de entrenamiento y de prueba
    onehot_features = data.get_onehot_dict()# se obtienen los datos de las variables categoricas 

    try:
        # populate storage with models
        # almacenar los modelos en firebase 
        trained_models = [] # lista de modelos que se han entrenado 
        for model_type in model_list:
            model = Model(model_type)
            model.build(X_train, y_train, onehot_features) # build model with data and onehot_features dict 
            # get the saved model in byte form
            pickle_bytes = model.pickle()
            # send it to firebase storage
            fb.send_pickle(
                bucket,
                pickle_bytes,
                fb.make_path(str(uid), str(proj_id), str(model_type)),
            )
            trained_models.append(model)
        # update firestore with descriptive stats (IQR)
        data.send_vars(proj_id, trained_models) # se envia los datos de las variables categoricas a firebase 
        return jsonify({"result": "success"}), 200 # se retorna un json con el resultado de la operacion en formato json
        
    
    except Exception as e:
        print(f"failed {e}")
        return (
            jsonify(
                {"result": "failure", "error": "520", "message": f"Build Failed: {e}"} 
                # error 520 es el codigo de error que se utiliza para indicar que la creacion del modelo ha fallado
            ),
            500,
        )


@app.route("/describe", methods=["POST"])
def describe():
    # this is a route for getting descriptive statistics about the dataframe
    # necessary to help users make informed decisions when creating models
    req_data = request.get_json()
    print(f"this is the reqdata {req_data}")
    # Brackets require these fields to be present
    # Sort of a safety contract to ensure we always have valid path
    proj_id = req_data["projId"]  # unique project hash
    newPath = req_data["csvPath"]
    data = Data.from_csv(newPath)
    description = data.pre_describe(proj_id)
    return jsonify(description)


if __name__ == "__main__":
    app.run()
