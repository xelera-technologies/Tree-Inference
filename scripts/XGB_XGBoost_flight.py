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
    import XlPluginDecisionTreeInference as xl

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

    modelFileName = "./XGB_Regression_Flight_" + str(number_of_trees) + "trees.pkl"
    modelFPGAFileName = "./XGB_Regression_Flight_" + str(number_of_trees) + "trees.xlmodel"


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

    if (os.path.isfile(modelFileName)):
        print("Model is already available. Training is NOT run")
    else:
        print("Model is not available, start training ...")

        start_time = time.perf_counter()
        model_regression.fit(x_train,y_train)
        print("Training_time:      ", time.perf_counter() - start_time, "s")
        pickle.dump(model_regression, open(modelFileName, 'wb'))


        xlSetup = xl.XlXGBSetup()
        num_classes = 1
        xlSetup.getModelForFPGA(model_regression.get_booster(), max_depth, num_classes, feature_names,modelFPGAFileName)
       

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
        dump_list = model_regression.get_booster().get_dump()
        number_of_trees = len(dump_list)
        print("SW Number of features:",x_test.shape[1])
        print("SW Number of trees:",number_of_trees)



    ######### Inference FPGA #######

    if (enable_FPGA_inference):

        print("Preparing HW inference ...")

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
            y_pred = xlrf.get_results()
        stop_time = time.perf_counter()
        hw_time_latency = stop_time - start_time


        mse=mean_squared_error(y_test, y_pred)
        print("HW mse",np.sqrt(mse))

        error = abs(y_test - y_pred)
        print("HW error",error.mean())
        print('HW predict latency (average on', nLoops,'runs for', y_pred.size, 'samples): ', "{:.2e}".format(hw_time_latency/nLoops), 's')
        print("HW predict throughput:", "{:.2e}".format(nLoops*y_pred.size/hw_time_throughput), "samples/s")
        dump_list = model_regression.get_booster().get_dump()
        number_of_trees = len(dump_list)
        print("HW Number of features:",x_test.shape[1])
        print("HW Number of trees:",number_of_trees)

        del xlrf    

############################
## Binomial
############################

if enable_binomial:

    print("##############################################")
    print("XGBoost Binomial with Numerical Features")

    modelFileName = "./XGB_Binomial_Flight_" + str(number_of_trees) + "trees.pkl"
    modelFPGAFileName = "./XGB_Binomial_Flight_" + str(number_of_trees) + "trees.xlmodel"

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

    if (os.path.isfile(modelFileName)):
        print("Model is already available. Training is NOT run")
    else:
        print("Model is not available, start training ...")

        start_time = time.perf_counter()
        model_binomial.fit(x_train,y_train)
        print("Training_time:      ", time.perf_counter() - start_time, "s")
        pickle.dump(model_binomial, open(modelFileName, 'wb'))


        xlSetup = xl.XlXGBSetup()
        num_classes = 2
        xlSetup.getModelForFPGA(model_binomial.get_booster(), max_depth, num_classes, feature_names, modelFPGAFileName)


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
        print("LOGINFO SW mse",np.sqrt(mse))

        error = abs(y_test - y_pred)
        print("SW error",error.mean())
        print('SW predict latency (average on', nLoops,'runs for', y_pred.size, 'samples): ', "{:.2e}".format(sw_time/nLoops), 's')
        print("SW predict throughput:", "{:.2e}".format(nLoops*y_pred.size/sw_time), "samples/s")
        dump_list = model_binomial.get_booster().get_dump()
        number_of_trees = len(dump_list)
        print("SW Number of features:",x_test.shape[1])
        print("SW Number of samples:", y_pred.size)
        print("SW Number of trees:",number_of_trees)


    ######### Inference FPGA #######

    if (enable_FPGA_inference):


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
            y_pred = xlrf.get_results()
        stop_time = time.perf_counter()
        hw_time_latency = stop_time - start_time
        y_pred = xlrf.xgb_binary_logistic(y_pred)


        mse=mean_squared_error(y_test, y_pred)
        print(" HW mse",np.sqrt(mse))

        error = abs(y_test - y_pred)
        print("HW error",error.mean())
        print('HW predict latency (average on', nLoops,'runs for', y_pred.size, 'samples): ', "{:.2e}".format(hw_time_latency/nLoops), 's')
        print("HW predict throughput:", "{:.2e}".format(nLoops*y_pred.size/hw_time_throughput), "samples/s")
        dump_list = model_binomial.get_booster().get_dump()
        number_of_trees = len(dump_list)
        print("HW Number of features:",x_test.shape[1])
        print("SW Number of samples:", y_pred.size)
        print("HW Number of trees:",number_of_trees)

        del xlrf


############################
## Multinomial
############################

if (enable_multinomial):

    print("##############################################")
    print("XGB Multinomial with Numerical Features")

    modelFileName = "./XGB_Multinomial_Flight_" + str(number_of_trees) + "trees.pkl"
    modelFPGAFileName = "./XGB_Multinomial_Flight_" + str(number_of_trees) + "trees.xlmodel"

    data = data_origin.copy()

    cols = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
    for item in cols:
        data[item] = data[item].astype("category").cat.codes

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

    print("XGB multinomial: start training ...")

    #set iterations= 500 to get good results
    model = xgb.XGBClassifier(objective="multi:softprob", booster="gbtree", random_state=42, max_depth=max_depth, n_estimators=max_number_of_trees, n_jobs=32)

    if (os.path.isfile(modelFileName)):
        print("Model is already available. Training is NOT run")
    else:
        print("Model is not available, start training ...")

        start_time = time.perf_counter()
        model.fit(x_train,y_train)
        print("Training_time:      ", time.perf_counter() - start_time, "s")
        pickle.dump(model, open(modelFileName, 'wb'))

        xlSetup = xl.XlXGBSetup()
        xlSetup.getModelForFPGA(model.get_booster(), max_depth, num_classes, feature_names, modelFPGAFileName)


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
            y_pred = model.predict_proba(x_test+i)
        stop_time = time.perf_counter()
        sw_time = stop_time - start_time


        y_pred = y_pred.argmax(1) + 1

        mse=mean_squared_error(y_test, y_pred)
        print("LOGINFO SW mse",np.sqrt(mse))

        error = abs(y_test - y_pred)
        print("LOGINFO SW error",error.mean())
        print('SW predict latency (average on', nLoops,'runs for', y_pred.size, 'samples): ', "{:.2e}".format(sw_time/nLoops), 's')
        print("SW predict throughput:", "{:.2e}".format(nLoops*y_pred.size/sw_time), "samples/s")
        dump_list = model.get_booster().get_dump()
        number_of_trees = len(dump_list)
        print("LOGINFO SW Number of features:",x_test.shape[1])
        print("LOGINFO SW Number of samples:", y_pred.size)
        print("LOGINFO SW Number of trees:",number_of_trees)

        del model
        del y_pred

    ######### Inference FPGA #######

    if (enable_FPGA_inference):

        try:
            f = open(modelFileName, 'rb')
            model = pickle.load(f)
        except IOError:
            print("File",modelFileName," not accessible")
        finally:
            f.close()

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
            y_pred = xlrf.get_results()
            y_pred = xlrf.xgb_softmax(y_pred)
        stop_time = time.perf_counter()
        hw_time_latency = stop_time - start_time

        y_pred = y_pred.argmax(1) + 1

        mse=mean_squared_error(y_test, y_pred)
        print("LOGINFO HW mse",np.sqrt(mse))

        error = abs(y_test - y_pred)
        print("LOGINFO HW error",error.mean())
        print('HW predict latency (average on', nLoops,'runs for', y_pred.size, 'samples): ', "{:.2e}".format(hw_time_latency/nLoops), 's')
        print("HW predict throughput:", "{:.2e}".format(nLoops*y_pred.size/hw_time_throughput), "samples/s")
        dump_list = model.get_booster().get_dump()
        number_of_trees = len(dump_list)
        print("LOGINFO HW Number of features:",x_test.shape[1])
        print("LOGINFO SW Number of samples:", y_pred.size)
        print("LOGINFO HW Number of trees:",number_of_trees)
       
        del model
        del y_pred
        del xlrf

if (enable_FPGA_inference):
    xl.shutdown()
