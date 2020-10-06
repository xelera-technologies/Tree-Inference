import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, roc_auc_score
import lightgbm as lgb
import time
import pickle
import os
import json
import sys

from utils import arguments

tree_inference_args = arguments.parse_arguments()

enable_regression = tree_inference_args.enable_regression
enable_binomial = tree_inference_args.enable_binomial
enable_multinomial = tree_inference_args.enable_multinomial


enable_SW_inference = tree_inference_args.enable_SW_inference
enable_FPGA_inference = tree_inference_args.enable_FPGA_inference


max_depth = tree_inference_args.max_depth
number_of_trees = tree_inference_args.number_of_trees
numTestSamples = tree_inference_args.num_test_samples
max_number_of_trees = number_of_trees


print("Loading dataset ...")
nLoops = tree_inference_args.n_loops
dataset_name = tree_inference_args.data_fpath
data_origin = pd.read_csv(dataset_name)

enable_categorical_features = False
cat_features_name = ["MONTH","DAY","DAY_OF_WEEK","AIRLINE","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]

data_origin = data_origin.sample(frac = 0.1, random_state=10)

data_origin = data_origin[["MONTH","DAY","DAY_OF_WEEK","AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT",
                 "ORIGIN_AIRPORT","AIR_TIME", "DEPARTURE_TIME","DISTANCE","ARRIVAL_DELAY"]]
data_origin.dropna(inplace=True)


############################
## Regression
############################

if enable_regression:

    print("################################################")
    if enable_categorical_features:
        print("LightGBM Regression with Categorical Features")
    else:
        print("LightGBM Regression with Numerical Features")

    modelFileName = "./LightGBM_Regression_Flight_" + str(number_of_trees) + "trees.pkl"
    modelFPGAFileName = "./LightGBM_Regression_Flight_" + str(number_of_trees) + "trees.xlmodel"


    data = data_origin.copy()

    cols = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
    for item in cols:
        data[item] = data[item].astype("category").cat.codes +1

    data_label = data["ARRIVAL_DELAY"]
    data = data.drop(["ARRIVAL_DELAY"], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(data, data_label,random_state=10, test_size=0.25)

    if numTestSamples <= y_test.shape[0]:
        x_test = x_test.head(numTestSamples)
        y_test = y_test.head(numTestSamples)
    else:
        numTestSamples = y_test.shape[0]
        print("numTestSamples capped to", numTestSamples)

    ######### Training ########

    print("Regression: Start training ...")

    #n_estimators = number of trees used
    model_regression = lgb.LGBMRegressor(learning_rate=0.15, num_leaves=900, max_depth=max_depth, n_estimators=max_number_of_trees)

    if (os.path.isfile(modelFileName)):
        print("Model is already available. Training is NOT run")
    else:
        print("Model is not available, start training ...")

        start_time = time.perf_counter()
        if enable_categorical_features:
            model_regression.fit(x_train,y_train, categorical_feature = cat_features_name)
        else:
            model_regression.fit(x_train,y_train)
        print("Training_time:      ", time.perf_counter() - start_time, "s")
        pickle.dump(model_regression, open(modelFileName, 'wb'))
        

        import XlPluginDecisionTreeInference as xl 
        xlSetup = xl.XlLightGBMSetup()
        bdump = model_regression.booster_.dump_model()
        jsonModelDump = json.dumps(bdump)
        xlSetup.getModelForFPGA(jsonModelDump, max_depth, modelFPGAFileName)

    ######### Inference SW ########

    if enable_SW_inference:

        print("Starting SW inference ...")

        try:
            f = open(modelFileName, 'rb')
            model_regression = pickle.load(f)
        except IOError:
            print("File",modelFileName," not accessible")
        finally:
            f.close()

        #throughput and latency measure
        start_time = time.perf_counter()
        for i in range(nLoops):
            y_pred = model_regression.predict(x_test+i)
        stop_time = time.perf_counter()
        sw_time = stop_time - start_time


        mse=mean_squared_error(y_test, y_pred)
        print("SW mse",np.sqrt(mse))

        error = abs(y_test - y_pred)
        print("SW error",error.mean())
        print('SW predict latency (average on', nLoops,'runs for', y_pred.size, 'samples): ', "{:.2e}".format(sw_time/nLoops), 's')
        print("SW predict throughput:", "{:.2e}".format(nLoops*y_pred.size/sw_time), "samples/s")
        print("SW Number of features:",x_test.shape[1])
        print("SW Number of trees:",number_of_trees)

    ######## Inference FPGA ############

    if enable_FPGA_inference:
        import XlPluginDecisionTreeInference as xl

        xlrf = xl.XlRFInference()
        xlrf.setModel(modelFPGAFileName)


        x_test_np = np.array(x_test, dtype=np.float32, order='C')

        print("Starting HW inference ...")
        #throughput measure
        start_time = time.perf_counter()
        for i in range(nLoops):
            xlrf.predict(x_test_np+i)
        for i in range(nLoops):
            y_pred = xlrf.get_results()
        stop_time = time.perf_counter()
        hw_time_throughput = stop_time - start_time

        #latency measure
        start_time = time.perf_counter()
        for i in range(nLoops):
            xlrf.predict(x_test_np+i)
        for i in range(nLoops):
            y_pred = xlrf.get_results()
        stop_time = time.perf_counter()
        hw_time_latency = stop_time - start_time


        mse=mean_squared_error(y_test, y_pred)
        print("HW mse",np.sqrt(mse))

        error = abs(y_test - y_pred)
        print("HW error",error.mean())
        print('HW predict latency (average on', nLoops,'runs for', y_pred.size, 'samples): ', "{:.2e}".format(hw_time_latency/nLoops), 's')
        print("HW predict throughput:", "{:.2e}".format(nLoops*y_pred.size/hw_time_throughput), "samples/s")
        dump_list = model_regression.booster_.dump_model()
        #number_of_trees = len(dump_list)
        print("HW Number of features:",x_test.shape[1])
        print("HW Number of trees:",number_of_trees)
        del xlrf



############################
## Binomial
############################
if (enable_binomial):

    print("##############################################")
    if enable_categorical_features:
        print("LightGBM Binomial with Categorical Features")
    else:
        print("LightGBM Binomial with Numerical Features")

    modelFileName = "./LightGBM_Binomial_Flight_" + str(number_of_trees) + "trees.pkl"
    modelFPGAFileName = "./LightGBM_Binomial_Flight_" + str(number_of_trees) + "trees.xlmodel"


    data = data_origin.copy()

    #create 2 classes for the label
    data['ARRIVAL_DELAY'] = np.where(data['ARRIVAL_DELAY'] <= 10, 0, data['ARRIVAL_DELAY'])
    data['ARRIVAL_DELAY'] = np.where(data['ARRIVAL_DELAY'] > 10, 1, data['ARRIVAL_DELAY'])


    data_label = data["ARRIVAL_DELAY"].astype("category").cat.codes
    data = data.drop(["ARRIVAL_DELAY"], axis=1)

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
        print("numTestSamples capped to", numTestSamples)

    ######### Training ########

    print("Start training ...")

    #set iterations= 500 to get good results
    model_binomial = lgb.LGBMClassifier(learning_rate=0.15, num_leaves=900, objective='binary', is_unbalance=True, max_depth=max_depth, n_estimators=max_number_of_trees)

    if (os.path.isfile(modelFileName)):
        print("Model is already available. Training is NOT run")
    else:
        print("Model is not available, start training ...")

        start_time = time.perf_counter()
        if enable_categorical_features:
            model_binomial.fit(x_train,y_train, categorical_feature = cat_features_name)
        else:
            model_binomial.fit(x_train,y_train)

        print("Training_time:      ", time.perf_counter() - start_time, "s")
        pickle.dump(model_binomial, open(modelFileName, 'wb'))


        bdump = model_binomial.booster_.dump_model()


        model_fpga_binomial = "no_model"

        import XlPluginDecisionTreeInference as xl 

        xlSetup = xl.XlLightGBMSetup()
        jsonModelDump = json.dumps(bdump)
        xlSetup.getModelForFPGA(jsonModelDump, max_depth, modelFPGAFileName)
       
            

    ######### Inference SW ########

    if enable_SW_inference:

        try:
            f = open(modelFileName, 'rb')
            model_binomial = pickle.load(f)
        except IOError:
            print("File",modelFileName," not accessible")
        finally:
            f.close()

        #throughput and latency measure
        start_time = time.perf_counter()
        for i in range(nLoops):
            y_pred = model_binomial.predict(x_test+i)
        stop_time = time.perf_counter()
        sw_time = stop_time - start_time

        mse=mean_squared_error(y_test, y_pred)
        print("SW mse",np.sqrt(mse))

        y_pred_binomial_sw = y_pred
        error = abs(y_test - y_pred)
        print("SW error",error.mean())
        print('SW predict latency (average on', nLoops,'runs for', y_pred.size, 'samples): ', "{:.2e}".format(sw_time/nLoops), 's')
        print("SW predict throughput:", "{:.2e}".format(nLoops*y_pred.size/sw_time), "samples/s")
        print("SW Number of features:",x_test.shape[1])
        del y_pred

    ######## Inference FPGA ############

    if enable_FPGA_inference:
        import XlPluginDecisionTreeInference as xl

        xlrf = xl.XlRFInference()
        xlrf.setModel(modelFPGAFileName)


        x_test_np = np.array(x_test, dtype=np.float32, order='C')

        print("Starting HW inference ...")
        #throughput measure
        start_time = time.perf_counter()
        for i in range(nLoops):
            xlrf.predict(x_test_np+i)
        for i in range(nLoops):
            y_pred = xlrf.get_results()
        stop_time = time.perf_counter()
        hw_time_throughput = stop_time - start_time

        #latency measure
        start_time = time.perf_counter()
        for i in range(nLoops):
            xlrf.predict(x_test_np+i)
        for i in range(nLoops):
            y_pred = xlrf.get_results()
        stop_time = time.perf_counter()
        hw_time_latency = stop_time - start_time

        y_pred_binomial_raw_hw = y_pred 
        y_pred = xlrf.xgb_binary_logistic(y_pred)

        y_pred_binomial_hw = y_pred


        mse=mean_squared_error(y_test, y_pred)
        print("HW mse",np.sqrt(mse))

        error = abs(y_test - y_pred)
        print("HW error",error.mean())
        print('HW predict latency (average on', nLoops,'runs for', y_pred.size, 'samples): ', "{:.2e}".format(hw_time_latency/nLoops), 's')
        print("HW predict throughput:", "{:.2e}".format(nLoops*y_pred.size/hw_time_throughput), "samples/s")
        dump_list = model_binomial.booster_.dump_model()
        print("HW Number of features:",x_test.shape[1])
        print("HW Number of trees:",number_of_trees)
        del model_binomial
        del model_fpga_binomial
        del y_pred
        del xlrf


############################
## Multinomial
############################

if enable_multinomial:

    print("##############################################")
    if enable_categorical_features:
        print("LightGBM Multinomial with Categorical Features")
    else:
        print("LightGBM Multinomial with Numerical Features")

    modelFileName = "./LightGBM_Multinomial_Flight_" + str(number_of_trees) + "trees.pkl"
    modelFPGAFileName = "./LightGBM_Multinomial_Flight_" + str(number_of_trees) + "trees.xlmodel"

    data = data_origin.copy()

    #create 4 classes for the label
    data['ARRIVAL_DELAY'] = np.where(data['ARRIVAL_DELAY'] <= 0, 0, data['ARRIVAL_DELAY'])
    data['ARRIVAL_DELAY'] = np.where((data['ARRIVAL_DELAY']>0 ) & (data['ARRIVAL_DELAY']<=5), 1, data['ARRIVAL_DELAY'])
    data['ARRIVAL_DELAY'] = np.where((data['ARRIVAL_DELAY']>5 ) & (data['ARRIVAL_DELAY']<=10), 2, data['ARRIVAL_DELAY'])
    data['ARRIVAL_DELAY'] = np.where(data['ARRIVAL_DELAY'] > 10, 3, data['ARRIVAL_DELAY'])

    num_classes = data['ARRIVAL_DELAY'].nunique()


    data_label = data["ARRIVAL_DELAY"].astype("category").cat.codes +1
    data = data.drop(["ARRIVAL_DELAY"], axis=1)

    cols = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
    for item in cols:
        data[item] = data[item].astype("category").cat.codes

    x_train, x_test, y_train, y_test = train_test_split(data, data_label,random_state=10, test_size=0.25)

    if numTestSamples <= y_test.shape[0]:
        x_test = x_test.head(numTestSamples)
        y_test = y_test.head(numTestSamples)
    else:
        numTestSamples = y_test.shape[0]
        print("LOGINFO numTestSamples capped to", numTestSamples)

    ######### Training ########

    if (os.path.isfile(modelFileName)):
        print("Model is already available. Training is NOT run")
    else:
        print("Model is not available, start training ...")

        #set iterations= 500 to get good results
        model_multinomial = lgb.LGBMClassifier( boosting_type='gbdt',learning_rate=0.15, num_leaves=900, objective='multiclass', max_depth=max_depth, n_estimators=max_number_of_trees, num_class=num_classes)

        start_time = time.perf_counter()
        if enable_categorical_features:
            # model.fit(x_train,y_train,cat_features_name)
            print("NOT supported")
        else:
            model_multinomial.fit(x_train,y_train)
            pickle.dump(model_multinomial, open(modelFileName, 'wb'))

        print("LOGINFO Training_time:      ", time.perf_counter() - start_time, "s")


        fpga_model_multinomial = "no model"
        import XlPluginDecisionTreeInference as xl
        xlSetup_multinomial = xl.XlLightGBMSetup()
        bdump = model_multinomial.booster_.dump_model()
        jsonModelDump = json.dumps(bdump)
        fpga_model_multinomial = xlSetup_multinomial.getModelForFPGA(jsonModelDump, max_depth, modelFPGAFileName)


    ######### Inference SW ########

    if enable_SW_inference:

        try:
            f = open(modelFileName, 'rb')
            model = pickle.load(f)
        except IOError:
            print("File",modelFileName," not accessible")
        finally:
            f.close()

        #throughput and latency measure
        start_time = time.perf_counter()
        for i in range(nLoops):
            y_pred = model.predict(x_test+i)
        stop_time = time.perf_counter()
        sw_time = stop_time - start_time



        print ("SW_pred = ", y_pred)
        mse=mean_squared_error(y_test, y_pred)
        print("SW mse",np.sqrt(mse))

        error = abs(y_test - y_pred)
        print("SW error",error.mean())
        print('SW predict latency (average on', nLoops,'runs for', y_pred.size, 'samples): ', "{:.2e}".format(sw_time/nLoops), 's')
        print("SW predict throughput:", "{:.2e}".format(nLoops*y_pred.size/sw_time), "samples/s")
        print("SW Number of features:",x_test.shape[1])


     ######### Inference FPGA #######

    if (enable_FPGA_inference):
        import XlPluginDecisionTreeInference as xl

        xlrf = xl.XlRFInference()
        xlrf.setModel(modelFPGAFileName)

        x_test_np = np.array(x_test, dtype=np.float32, order='C')

        print("Starting HW inference ...")
        #throughput measure
        start_time = time.perf_counter()
        for i in range(nLoops):
            xlrf.predict(x_test_np+i)
        for i in range(nLoops):
            y_pred = xlrf.get_results()
            y_pred = xlrf.xgb_softmax(y_pred)
        stop_time = time.perf_counter()
        hw_time_throughput = stop_time - start_time

        #latency measure
        start_time = time.perf_counter()
        for i in range(nLoops):
            xlrf.predict(x_test_np+i)
        for i in range(nLoops):
            y_pred = xlrf.get_results()
            y_pred = xlrf.xgb_softmax(y_pred)
        stop_time = time.perf_counter()
        hw_time_latency = stop_time - start_time

        y_pred = y_pred.argmax(1) +1

        print("HW_pred = ", y_pred)

        mse=mean_squared_error(y_test, y_pred)
        print("LOGINFO HW mse",np.sqrt(mse))

        error = abs(y_test - y_pred)
        print("LOGINFO HW error",error.mean())
        print('HW predict latency (average on', nLoops,'runs for', y_pred.size, 'samples): ', "{:.2e}".format(hw_time_latency/nLoops), 's')
        print("HW predict throughput:", "{:.2e}".format(nLoops*y_pred.size/hw_time_throughput), "samples/s")
        print("LOGINFO HW Number of features:",x_test.shape[1])
        print("LOGINFO SW Number of samples:", y_pred.size)
        print("LOGINFO HW Number of trees:",number_of_trees)
        print("LOGINFO HW Nuber of classes: ", num_classes)

        del xlrf   

if (enable_FPGA_inference):
    xl.shutdown()

