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



tree_inference_args = arguments.parse_arguments()

enable_regression = tree_inference_args.enable_regression
enable_binomial = tree_inference_args.enable_binomial
enable_multinomial = tree_inference_args.enable_multinomial

enable_SW_inference = tree_inference_args.enable_SW_inference
enable_FPGA_inference = tree_inference_args.enable_FPGA_inference

max_depth = tree_inference_args.max_depth
number_of_trees = tree_inference_args.number_of_trees
max_number_of_trees = number_of_trees
numTestSamples = tree_inference_args.num_test_samples

nLoops = tree_inference_args.n_loops

print("Loading dataset ...")
dataset_name = tree_inference_args.data_fpath
data_origin = pd.read_csv(dataset_name)
if (enable_FPGA_inference):
    import XlPluginRandomForest as xl

data_origin = data_origin.sample(frac = 0.1, random_state=10)


feature_names = ["MONTH","DAY","DAY_OF_WEEK","AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT",
                 "ORIGIN_AIRPORT","AIR_TIME", "DEPARTURE_TIME","DISTANCE","ARRIVAL_DELAY"]

data_origin = data_origin[feature_names]
data_origin.dropna(inplace=True)

feature_names.remove("ARRIVAL_DELAY") # we do not want this to be considered 



############################
## Regression
############################

if enable_regression:

    print("################################################")
    print("XGB Regression with Numerical Features")


    data = data_origin.copy()

    cols = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
    for item in cols:
        data[item] = data[item].astype("category").cat.codes +1

  #  data['ARRIVAL_DELAY'] = np.where(data['ARRIVAL_DELAY'] < 0, 0, data['ARRIVAL_DELAY'])

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

    print("Start training ...")

    #set "n_estimators": 300 to get good results
    model_regression = xgb.XGBRegressor(objective="reg:squarederror", booster="gbtree", random_state=42, max_depth=max_depth, n_estimators=max_number_of_trees, base_score=0.0, n_jobs = 32)

    start_time = time.perf_counter()
    model_regression.fit(x_train,y_train)
    print("Training_time:      ", time.perf_counter() - start_time, "s")

    print('model_regression = ', model_regression)

    if (enable_FPGA_inference):
        xlSetup = xl.XlXGBSetup()
        num_classes = 1
        fpga_model_regression = xlSetup.getModelForFPGA(model_regression.get_booster(), max_depth, num_classes, feature_names)



    ######### Inference SW ########

    if enable_SW_inference:

        print("Starting SW inference ...")
        sw_time = 0
        for i in range(nLoops):
            sw_start_time = time.perf_counter()
            y_pred = model_regression.predict(x_test)
            sw_stop_time = time.perf_counter()
            sw_time += (sw_stop_time - sw_start_time)
        sw_time = sw_time/nLoops

        mse=mean_squared_error(y_test, y_pred)
        print("SW mse",np.sqrt(mse))


        error = abs(y_test - y_pred)
        print("SW error",error.mean())
        print('SW predict time (average on', nLoops,'runs for', y_pred.size, 'samples): ', sw_time, 's')
        dump_list = model_regression.get_booster().get_dump()
        number_of_trees = len(dump_list)
        print("SW Number of features:",x_test.shape[1])
        print("SW Number of trees:",number_of_trees)



    ######### Inference FPGA #######

    if (enable_FPGA_inference):

        print("Preparing HW inference ...")

        inference_engine = xl.XlRFInference()
        inference_engine.setModel(fpga_model_regression)

        x_test_np = np.array(x_test, dtype=np.float32, order='C')

        print("Starting HW inference ...")

        hw_start_time = time.perf_counter()
        for i in range(nLoops):
            inference_engine.predict(x_test_np)
        for i in range(nLoops):
            y_pred = inference_engine.get_results()
        hw_stop_time = time.perf_counter()
        hw_time = (hw_stop_time - hw_start_time)/nLoops



        mse=mean_squared_error(y_test, y_pred)
        print("HW mse",np.sqrt(mse))

        error = abs(y_test - y_pred)
        print("HW error",error.mean())
        print('HW predict time (average on', nLoops,'runs for', y_pred.size, 'samples): ', hw_time, 's')
        dump_list = model_regression.get_booster().get_dump()
        number_of_trees = len(dump_list)
        print("HW Number of features:",x_test.shape[1])
        print("HW Number of trees:",number_of_trees)

        del inference_engine    

############################
## Binomial
############################

if enable_binomial:

    print("##############################################")
    print("XGBoost Binomial with Numerical Features")

    data = data_origin.copy()

    #create 2 classes for the label
    data['ARRIVAL_DELAY'] = np.where(data['ARRIVAL_DELAY'] <= 10, 0, data['ARRIVAL_DELAY'])
    data['ARRIVAL_DELAY'] = np.where(data['ARRIVAL_DELAY'] > 10, 1, data['ARRIVAL_DELAY'])


    data_label = data["ARRIVAL_DELAY"].astype("category").cat.codes
    data = data.drop(["ARRIVAL_DELAY"], axis=1)

    print(data.head(5) )

    cols = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
    for item in cols:
        data[item] = data[item].astype("category").cat.codes +1

    x_train, x_test, y_train, y_test = train_test_split(data, data_label,
                                                    random_state=10, test_size=0.25)

    if numTestSamples <= y_test.shape[0]:
        x_test = x_test.head(numTestSamples)
        y_test = y_test.head(numTestSamples)
    else:
        numTestSamples = y_test.shape[0]
        print("LOGINFO numTestSamples capped to", numTestSamples)

    ######### Training ########

    print("XGB Binomial: start training ...")

    #set iterations= 500 to get good results
    model_binomial = xgb.XGBClassifier(objective="binary:logistic", booster="gbtree", random_state=42, max_depth=max_depth, n_estimators=max_number_of_trees, n_jobs=32)

    start_time = time.perf_counter()
    model_binomial.fit(x_train,y_train)
    print("Training_time:      ", time.perf_counter() - start_time, "s")


    print('model_binomial = ', model_binomial)

    if (enable_FPGA_inference):
        xlSetup = xl.XlXGBSetup()
        num_classes = 2
        fpga_model_binomial = xlSetup.getModelForFPGA(model_binomial.get_booster(), max_depth, num_classes, feature_names)



    ######### Inference SW ########

    if enable_SW_inference:

        sw_time = 0
        for i in range(nLoops):
            sw_start_time = time.perf_counter()
            y_pred = model_binomial.predict(x_test)
            sw_stop_time = time.perf_counter()
            sw_time += (sw_stop_time - sw_start_time)

        sw_time = sw_time/nLoops


        mse=mean_squared_error(y_test, y_pred)
        print("LOGINFO SW mse",np.sqrt(mse))

        error = abs(y_test - y_pred)
        print("SW error",error.mean())
        print('SW predict time (average on', nLoops,'runs for', y_pred.size, 'samples): ', sw_time, 's')
        dump_list = model_binomial.get_booster().get_dump()
        number_of_trees = len(dump_list)
        print("SW Number of features:",x_test.shape[1])
        print("SW Number of samples:", y_pred.size)
        print("SW Number of trees:",number_of_trees)


    ######### Inference FPGA #######

    if (enable_FPGA_inference):


        inference_engine = xl.XlRFInference()
        inference_engine.setModel(fpga_model_binomial)

        x_test_np = np.array(x_test, dtype=np.float32, order='C')

        hw_start_time = time.perf_counter()
        for i in range(nLoops):
            inference_engine.predict(x_test_np)
        for i in range(nLoops):
            y_pred = inference_engine.get_results()
        hw_stop_time = time.perf_counter()
        hw_time = (hw_stop_time - hw_start_time)/nLoops

        y_pred = inference_engine.xgb_binary_logistic(y_pred)


        mse=mean_squared_error(y_test, y_pred)
        print(" HW mse",np.sqrt(mse))

        error = abs(y_test - y_pred)
        print("HW error",error.mean())
        print('HW predict time (average on', nLoops,'runs for', y_pred.size, 'samples): ', hw_time, 's')
        dump_list = model_binomial.get_booster().get_dump()
        number_of_trees = len(dump_list)
        print("HW Number of features:",x_test.shape[1])
        print("SW Number of samples:", y_pred.size)
        print("HW Number of trees:",number_of_trees)

        del inference_engine

if (enable_multinomial):
    print("\n multinomial classification currently not supported with XGBoost")           

if (enable_FPGA_inference):
    xl.shutdown()
