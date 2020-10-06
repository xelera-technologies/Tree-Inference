import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, roc_auc_score
import xgboost as xgb
import time
import pickle
import os
import sys
from xgboost import DMatrix as DMatrix
from utils import arguments

import XlPluginDecisionTreeInference as xl

tree_inference_args = arguments.parse_arguments()


######Free parameters######
max_depth = tree_inference_args.max_depth
number_of_trees = tree_inference_args.number_of_trees
numTestSamples = tree_inference_args.num_test_samples
nLoops = tree_inference_args.n_loops
dataset_name = tree_inference_args.data_fpath

modelFileName = "./XGB_Regression_Flight_" + str(number_of_trees) + "trees.pkl"
modelFPGAFileName = "./XGB_Regression_Flight_" + str(number_of_trees) + "trees.xlmodel"
##############################

print("Loading dataset ...")

data_origin = pd.read_csv(dataset_name)
data_origin = data_origin.sample(frac = 0.1, random_state=10)

feature_names = ["MONTH","DAY","DAY_OF_WEEK","AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT",
                 "ORIGIN_AIRPORT","AIR_TIME", "DEPARTURE_TIME","DISTANCE","ARRIVAL_DELAY"]

data_origin = data_origin[feature_names]
data_origin.dropna(inplace=True)

feature_names.remove("ARRIVAL_DELAY") # its the target

num_classes = 1
max_depth = 8

data = data_origin.copy()

cols = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
for item in cols:
    data[item] = data[item].astype("category").cat.codes +1


data_label = data["ARRIVAL_DELAY"]
data = data.drop(["ARRIVAL_DELAY"], axis=1)

print(data.head(5) )

x_train, x_test, y_train, y_test = train_test_split(data, data_label,random_state=10, test_size=0.25)

if numTestSamples <= y_test.shape[0]:
    x_test = x_test.head(numTestSamples)
    y_test = y_test.head(numTestSamples)
else:
    numTestSamples = y_test.shape[0]

######### Training ########

if (os.path.isfile(modelFileName)):
    print("Model is already available. Training is NOT run")
else:
    print("Model is not available, start training ...")

    model = xgb.XGBRegressor(objective="reg:squarederror", booster="gbtree", random_state=42, max_depth=max_depth, n_estimators=number_of_trees, base_score=0.0, n_jobs = 32)
    model.fit(x_train,y_train)

    # save the trained xgboost model
    pickle.dump(model, open(modelFileName, 'wb'))

    del model


if (os.path.isfile(modelFPGAFileName)):
    print ("FPGA Model is already available. xlmodel conversion is NOT run")
else:
    print("FPGA Model is not available, running xlmodel conversion ...")

    try:
        f = open(modelFileName, 'rb')
        model = pickle.load(f)
    except IOError:
        print("File",modelFileName," not accessible")
    finally:
        f.close()

    #convert model for FPGA
    xlSetup = xl.XlXGBSetup()
    fpga_model = xlSetup.getModelForFPGA(model.get_booster(), max_depth, num_classes, feature_names, modelFPGAFileName)

    del xlSetup
    del fpga_model
    del model

######### Inference ########
print("Running Inference ...")

try:  
    inference_engine = xl.XlRFInference()
    inference_engine.setModel(modelFPGAFileName)
except:
    print("File",modelFPGAFileName," not accessible")

start_time = time.perf_counter()
x_test_np = np.array(x_test, dtype=np.float32, order='C')
for i in range(nLoops):
    # To avoid caching and to have more accurate performance numbers, pass a new array every time 
    inference_engine.predict(x_test_np + i)
for i in range(nLoops):
    y_pred = inference_engine.get_results()
stop_time = time.perf_counter()
execution_time = stop_time-start_time

mse=mean_squared_error(y_test, y_pred)
del inference_engine
xl.shutdown()


##############################
print("INFO: Number of features:",x_test.shape[1])
print("INFO: Number of samples:", y_pred.size)
print("INFO: Number of trees:",number_of_trees)

print("INFO: FPGA mse:",np.sqrt(mse))

print("INFO: FPGA throughput:", "{:.2e}".format(nLoops*y_pred.size/execution_time), "samples/s")


