import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, roc_auc_score
from sklearn.ensemble import RandomForestClassifier as skrfc
from sklearn.ensemble import RandomForestRegressor as skrfr
import pickle
import os
import time
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

nLoops = tree_inference_args.n_loops

print("Loading dataset...")

dataset_name = tree_inference_args.data_fpath
data_origin = pd.read_csv(dataset_name)
if (enable_FPGA_inference):
    import XlPluginRandomForest as xl

data_origin = data_origin.sample(frac = 0.1, random_state=10)

data_origin = data_origin[["MONTH","DAY","DAY_OF_WEEK","AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT",
                 "ORIGIN_AIRPORT","AIR_TIME", "DEPARTURE_TIME","DISTANCE","ARRIVAL_DELAY"]]
data_origin.dropna(inplace=True)


data_origin = data_origin.head(300000)





############################
## Regression
############################

if enable_regression:

    print("################################################")
    print("RF Regression with Numerical Features")

    data = data_origin.copy()


    cols = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
    for item in cols:
        data[item] = data[item].astype("category").cat.codes +1


    data_label = data["ARRIVAL_DELAY"]
    data = data.drop(["ARRIVAL_DELAY"], axis=1)

    data = pd.DataFrame(data.astype(np.float32))
    # cuML Random Forest Classifier requires the labels to be integers
    data_label = pd.Series(data_label.astype(np.int32))


    x_train, x_test, y_train, y_test = train_test_split(data, data_label,random_state=10, test_size=0.25)

    if numTestSamples <= y_test.shape[0]:
        x_test = x_test.head(numTestSamples)
        y_test = y_test.head(numTestSamples)
    else:
        numTestSamples = y_test.shape[0]
        print("numTestSamples capped to", numTestSamples)


    ######### Training ########

    #set "n_estimators": 300 to get good results
    model_sk_regression = skrfr(max_depth=max_depth,n_estimators=number_of_trees, n_jobs=32, random_state=1234)
    start_time = time.perf_counter()
    model_sk_regression.fit(x_train, y_train)

    print("Training_time:      ", time.perf_counter() - start_time, "s")


    if enable_FPGA_inference:
        import XlPluginRandomForest as xl
        start_time = time.perf_counter()
        xlrfsetup = xl.XlRandomForestSetup()
        model_xlrf_regression = xlrfsetup.getModelForFPGA(model_sk_regression)
        del xlrfsetup
        print("HW_formatting_time:      ", time.perf_counter() - start_time, "s")



    ######### Inference SW ########

    if enable_SW_inference:
        
        sw_time = 0
        print("Starting SW inference...")
        
        for i in range(nLoops):
            sw_start_time = time.perf_counter()
            y_pred = model_sk_regression.predict(x_test)
            sw_stop_time = time.perf_counter()
            sw_time += (sw_stop_time - sw_start_time)

        sw_time = sw_time/nLoops

        mse=mean_squared_error(y_test, y_pred)

        print("SW mse",np.sqrt(mse))
        print(y_test)
        print(y_pred)
        error = abs(y_test - y_pred)
        print("SW error",error.mean())
        print('SW predict time (average on', nLoops,'runs for', y_pred.size, 'samples): ', sw_time, 's')
        print("SW Number of features:",x_test.shape[1])
        print("SW Number of trees:",number_of_trees)


    ######### Inference FPGA ########

    if (enable_FPGA_inference):
        import XlPluginRandomForest as xl
        print("Preparing HW inference...")


        setup_starttime = time.perf_counter()

        xlrf = xl.XlRFInference()
        xlrf.setModel(model_xlrf_regression)

        setup_endtime = time.perf_counter()
        HW_setup_time = setup_endtime - setup_starttime

        x_nd = np.array(x_test, dtype=np.float32, order='C')

        print("Starting HW Inference...")
        hw_time = 0
        for i in range(nLoops):
            hw_start_time = time.perf_counter()
            y_pred = xlrf.predict(x_nd)
            hw_stop_time = time.perf_counter()
            hw_time += (hw_stop_time - hw_start_time)

        hw_time =  hw_time/nLoops

        mse=mean_squared_error(y_test, y_pred)
        print("HW mse",np.sqrt(mse))
        error = abs(y_test - y_pred)
        print("HW error",error.mean())
        print('HW predict time (average on', nLoops,'runs for', y_pred.size, 'samples): ', hw_time, 's')
        if enable_SW_inference:
            print('HW vs.SW speedup: ', sw_time/hw_time, 'X')
        print("HW Number of features:",x_test.shape[1])
        print("HW Number of trees:",number_of_trees)
        del xlrf

############################
## Binomial
############################

if enable_binomial:

    print("##############################################")
    print("RF Binomial with Numerical Features")

    data = data_origin.copy()

    cols = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
    for item in cols:
        data[item] = data[item].astype("category").cat.codes +1

    #create 2 classes for the label
    data['ARRIVAL_DELAY'] = np.where(data['ARRIVAL_DELAY'] <= 10, 0, data['ARRIVAL_DELAY'])
    data['ARRIVAL_DELAY'] = np.where(data['ARRIVAL_DELAY'] > 10, 1, data['ARRIVAL_DELAY'])

    data_label = data["ARRIVAL_DELAY"]
    data = data.drop(["ARRIVAL_DELAY"], axis=1)

    data = pd.DataFrame(data.astype(np.float32))
    # cuML Random Forest Classifier requires the labels to be integers
    data_label = pd.Series(data_label.astype(np.int32))

    x_train, x_test, y_train, y_test = train_test_split(data, data_label,random_state=10, test_size=0.25)

    if numTestSamples <= y_test.shape[0]:
        x_test = x_test.head(numTestSamples)
        y_test = y_test.head(numTestSamples)
    else:
        numTestSamples = y_test.shape[0]
        print("numTestSamples capped to", numTestSamples)


    ######### Training ########

    print("SK binomial: Start training ...")

    #set iterations= 500 to get good results
    model_sk_binomial = skrfc(max_depth=max_depth,n_estimators=number_of_trees, n_jobs=32, random_state=1234)

    start_time = time.perf_counter()
    model_sk_binomial.fit(x_train,y_train)
    print("Training_time:      ", time.perf_counter() - start_time, "s")

    if enable_FPGA_inference:
        import XlPluginRandomForest as xl

        start_time = time.perf_counter()
        xlrfsetup = xl.XlRandomForestSetup()
        model_xlrf_binomial = xlrfsetup.getModelForFPGA(model_sk_binomial)
        del xlrfsetup
        print("HW_formatting_time:      ", time.perf_counter() - start_time, "s")


    ######### Inference SW ########

    if enable_SW_inference:

        print("Starting SW inference ...")

        sw_time = 0
        for i in range(nLoops):
            sw_start_time = time.perf_counter()
            y_pred = model_sk_binomial.predict(x_test)
            sw_stop_time = time.perf_counter()
            sw_time += (sw_stop_time - sw_start_time)

        sw_time = sw_time/nLoops

        mse=mean_squared_error(y_test, y_pred)

        print("SW mse",np.sqrt(mse))
        error = abs(y_test - y_pred)
        print("SW error",error.mean())
        print("SW accuracy score",accuracy_score(y_test, y_pred))
        print('SW predict time (average on', nLoops,'runs for', y_pred.size, 'samples): ', sw_time, 's')
        print("SW Number of features:",x_test.shape[1])
        print("SW Number of trees:",number_of_trees)
        print("SW Number of classes:",model_sk_binomial.classes_.shape[0])


    ######### Inference FPGA ########

    if (enable_FPGA_inference):
        import XlPluginRandomForest as xl
        print("Preparing HW inference ...")


        setup_starttime = time.perf_counter()

        xlrf = xl.XlRFInference()
        xlrf.setModel(model_xlrf_binomial)

        setup_endtime = time.perf_counter()
        HW_setup_time = setup_endtime - setup_starttime

        x_nd = np.array(x_test, dtype=np.float32, order='C')
        print("Starting HW inference ...")
        hw_time = 0
        for i in range(nLoops):
            hw_start_time = time.perf_counter()
            y_pred = xlrf.predict(x_nd)
            hw_stop_time = time.perf_counter()
            hw_time += (hw_stop_time - hw_start_time)

        hw_time =  hw_time / nLoops

        y_pred = y_pred.argmax(1)
        y_pred = model_sk_binomial.classes_.take(y_pred, axis=0)

        y_test = y_test.astype(int)


        mse=mean_squared_error(y_test, y_pred)
        print("HW mse",np.sqrt(mse))
        error = abs(y_test - y_pred)
        print("HW error",error.mean())
        print("HW accuracy score",accuracy_score(y_test, y_pred))
        print('HW predict time (average on', nLoops,'runs for', y_pred.size, 'samples): ', hw_time, 's')
        if enable_SW_inference:
            print('HW vs.SW speedup: ', sw_time/hw_time, 'X')
        print("HW Number of features:",x_test.shape[1])
        print("HW Number of trees:",number_of_trees)
        print("HW Number of classes:",model_sk_binomial.classes_.shape[0])
        
        del xlrf


############################
## Multinomial
############################

if enable_multinomial:

    print("##############################################")
    print("RF Multinomial with Numerical Features")


    data = data_origin.copy()

    cols = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
    for item in cols:
        data[item] = data[item].astype("category").cat.codes

    #create 10 classes for the label
    data['ARRIVAL_DELAY'] = np.where(data['ARRIVAL_DELAY'] <= 0, 0, data['ARRIVAL_DELAY'])
    data['ARRIVAL_DELAY'] = np.where((data['ARRIVAL_DELAY']>0 ) & (data['ARRIVAL_DELAY']<=1), 1, data['ARRIVAL_DELAY'])
    data['ARRIVAL_DELAY'] = np.where((data['ARRIVAL_DELAY']>1 ) & (data['ARRIVAL_DELAY']<=2), 2, data['ARRIVAL_DELAY'])
    data['ARRIVAL_DELAY'] = np.where((data['ARRIVAL_DELAY']>2 ) & (data['ARRIVAL_DELAY']<=3), 3, data['ARRIVAL_DELAY'])
    data['ARRIVAL_DELAY'] = np.where((data['ARRIVAL_DELAY']>3 ) & (data['ARRIVAL_DELAY']<=4), 4, data['ARRIVAL_DELAY'])
    data['ARRIVAL_DELAY'] = np.where((data['ARRIVAL_DELAY']>4 ) & (data['ARRIVAL_DELAY']<=5), 5, data['ARRIVAL_DELAY'])
    data['ARRIVAL_DELAY'] = np.where((data['ARRIVAL_DELAY']>5 ) & (data['ARRIVAL_DELAY']<=6), 6, data['ARRIVAL_DELAY'])
    data['ARRIVAL_DELAY'] = np.where((data['ARRIVAL_DELAY']>6 ) & (data['ARRIVAL_DELAY']<=7), 7, data['ARRIVAL_DELAY'])
    data['ARRIVAL_DELAY'] = np.where((data['ARRIVAL_DELAY']>7 ) & (data['ARRIVAL_DELAY']<=8), 8, data['ARRIVAL_DELAY'])
    data['ARRIVAL_DELAY'] = np.where(data['ARRIVAL_DELAY'] > 8, 9, data['ARRIVAL_DELAY'])



    data_label = data["ARRIVAL_DELAY"] + 0
    # data_label = data["ARRIVAL_DELAY"].astype("category").cat.codes +1
    data = data.drop(["ARRIVAL_DELAY"], axis=1)

    data = pd.DataFrame(data.astype(np.float32))
    data_label = pd.Series(data_label.astype(np.int32))

    x_train, x_test, y_train, y_test = train_test_split(data, data_label,random_state=10, test_size=0.25)

    if numTestSamples <= y_test.shape[0]:
        x_test = x_test.head(numTestSamples)
        y_test = y_test.head(numTestSamples)
    else:
        numTestSamples = y_test.shape[0]
        print("numTestSamples capped to", numTestSamples)

    ######### Training ########

    print("SK multinomial: Start training ...")

    #set iterations= 500 to get good results
    model_sk_multinomial = skrfc(max_depth=max_depth,n_estimators=number_of_trees, n_jobs=32, random_state=1234)

    start_time = time.perf_counter()
    model_sk_multinomial.fit(x_train,y_train)
    print("Training_time:      ", time.perf_counter() - start_time, "s")

    if enable_FPGA_inference:
        import XlPluginRandomForest as xl


        start_time = time.perf_counter()
        xlrfsetup = xl.XlRandomForestSetup()
        model_xlrf_multinomial = xlrfsetup.getModelForFPGA(model_sk_multinomial)
        del xlrfsetup
        print("HW_formatting_time:      ", time.perf_counter() - start_time, "s")


    ######### Inference SW ########

    if enable_SW_inference:

        print("Starting SW inference ...")

        sw_time = 0
        for i in range(nLoops):
            sw_start_time = time.perf_counter()
            y_pred = model_sk_multinomial.predict(x_test)
            sw_stop_time = time.perf_counter()
            sw_time += (sw_stop_time - sw_start_time)

        sw_time = sw_time / nLoops

        mse=mean_squared_error(y_test, y_pred)

        print("SW mse",np.sqrt(mse))
        error = abs(y_test - y_pred)
        print("SW error",error.mean())
        print("SW accuracy score",accuracy_score(y_test, y_pred))
        print('SW predict time (average on', nLoops,'runs for', y_pred.size, 'samples): ', sw_time, 's')
        print("SW Number of features:",x_test.shape[1])
        print("SW Number of trees:",number_of_trees)
        print("SW Number of classes:",model_sk_multinomial.classes_.shape[0])
    
    ######### Inference FPGA ########

    if (enable_FPGA_inference):
        import XlPluginRandomForest as xl

        print("Preparing HW inference ...")

        setup_starttime = time.perf_counter()

        xlrf = xl.XlRFInference()
        xlrf.setModel(model_xlrf_multinomial)

        setup_endtime = time.perf_counter()
        HW_setup_time = setup_endtime - setup_starttime

        x_nd = np.array(x_test, dtype=np.float32, order='C')

        print("Starting HW inference ...")
        hw_time = 0
        for i in range(nLoops):
            hw_start_time = time.perf_counter()           
            y_pred = xlrf.predict(x_nd)
            hw_stop_time = time.perf_counter()
            hw_time += (hw_stop_time - hw_start_time)

        hw_time =  hw_time / nLoops

        y_pred = y_pred.argmax(1)
        y_pred = model_sk_multinomial.classes_.take(y_pred, axis=0)

        y_test = np.array(y_test)
        y_test = y_test.astype(int)


        mse=mean_squared_error(y_test, y_pred)
        print("HW mse",np.sqrt(mse))
        error = abs(y_test - y_pred)
        print("HW error",error.mean())
        print("HW accuracy score",accuracy_score(y_test, y_pred))
        print('HW predict time (average on', nLoops,'runs for', y_pred.size, 'samples): ', hw_time, 's')
        if enable_SW_inference:
            print('HW vs.SW speedup: ', sw_time/hw_time, 'X')
        print("HW Number of features:",x_test.shape[1])
        print("HW Number of trees:",number_of_trees)
        print("HW Number of classes:",model_sk_multinomial.classes_.shape[0])
        del xlrf

if (enable_FPGA_inference):
    xl.shutdown()
