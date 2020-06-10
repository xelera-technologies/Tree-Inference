# Predict the flight delay

This usage example shows how to predict a flight delay using the [2015 Flight Delays and Cancellations](https://www.kaggle.com/usdot/flight-delays?select=flights.csv) dataset.

#### Prerequisites

* Install and run the Xelera decision Tree Inference AMI or Docker Image
* [Download](https://www.kaggle.com/usdot/flight-delays/data?select=flights.csv) the flight delay public dataset
* Load the file `flight.csv`:
    * AMI: `scp -i <certificate_name.pem> flight.csv centos@<EC2_IP_address>:/home/centos/xelera-demo/data/flight-delays/flights.csv`
    * Docker:
        * Get the running `<containers_id>`: `docker ps | grep "xtil"`
        * `docker cp flight.csv <container_id>:/app/xelera_demo/data/flight-delays/flights.csv`
* Get a copy of the example scripts provided by Xelera decision Tree Inference GitHub repository:
    1. Navigate to the `/app/xelera_demo` folder.
    2. Clone the Xelera decision Tree Inference GitHub repository: `git clone https://github.com/xelera-technologies/Tree-Inference.git`

#### Quick Start

1. Navigate to the folder `/app/xelera_demo/Tree-Inference/`
2. From inside the Tree-Inference folder run the experiment: `python3 scripts/x_y_z.py [args]` where `x` is the algorithm name, `y` is the framework name and `z` is the dataset name. Available are:

| Algorithm | Framework | Dataset | script name |
| --------- | --------- | ------- | ----------- |
| XGB       | XGBoost   | flight-delays | XGB_XGBoost_flight.py |
| LightGBM  | LightGBM  | flight-delays | LightGBM_LightGBM_flight.py |
| Random Forest  | scikit-learn | flight-delays | RF_scikit_flight.py |

See the following table for experiment arguments:

| argument | values | description | default value |
| -------- | ---- | ----------- | --------------- |
| --enable_SW_inference bool | true or false | whether to perform a CPU-based inference | true |
| --enable_FPGA_inference bool | true or false | whether to perform a FPGA-based inference | true |
| --enable_regression bool | true or false | evaluate regression experiment | true |
| --enable_binomial bool | true or false | evaluate binomial classification experiment | false |
| --enable_multinomial bool | true or false | evaluate multinomial classification experiment | false |
| --max_depth int | integer | the maximum depth of decision trees | 8 |
| --num_test_samples int | integer | the number of samples to infer | 100 |
| --number_of_trees int | integer | the number of trees to use for the experiment | 100 |
| --n_loops int | integer | the number of iterations over which the execution time is averaged | 1000 |
| --data_fpath string | string | file path to the dataset |

###### Random Forest multinomial classification inference on SW and HW (example)

**Docker**:
Run the Random Forest multinomial classification with 100 trees and 1000 samples using `python3 scripts/RF_scikit_flight.py --data_fpath /app/xelera_demo/data/flight-delays/flights.csv --enable_multinomial true --enable_regression false --number_of_trees 100 --num_test_samples 1000`. You will be prompted the accurracy and latency measures for CPU (SW) and FPGA (HW) inference runs. Note that training and execution in software might take some time.

**AWS**:
Run the Random Forest multinomial classification with 100 trees and 1000 samples using `python3 scripts/RF_scikit_flight.py --data_fpath /home/centos/xelera-demo/data/flight-delays/flights.csv --enable_multinomial true --enable_regression false --number_of_trees 100 --num_test_samples 1000`. You will be prompted the accurracy and latency measures for CPU (SW) and FPGA (HW) inference runs. Note that training and execution in software might take some time.

###### Random Forest multinomial classification inference on HW only (example)
**Docker**:
Run the Random Forest multinomial classification with 100 trees and 1000 samples using `python3 scripts/RF_scikit_flight.py --data_fpath /app/xelera_demo/data/flight-delays/flights.csv --enable_SW_inference false --enable_multinomial true --enable_regression false --number_of_trees 100 --num_test_samples 1000`. You will be prompted the accurracy and latency measures for FPGA (HW) inference runs.

**AWS**:
Run the Random Forest multinomial classification with 100 trees and 1000 samples using `python3 scripts/RF_scikit_flight.py --data_fpath /home/centos/xelera-demo/data/flight-delays/flights.csv --enable_SW_inference false --enable_multinomial true --enable_regression false --number_of_trees 100 --num_test_samples 1000`. You will be prompted the accurracy and latency measures for FPGA (HW) inference runs.


#### Step-by-Step Guide

You will be guided through an example which shows you how to leverage the Xelera Tree Inference Engine. It will cover the setup phase, in which a trained model from one of the supported frameworks is transformed into a model that can be loaded to the FPGA, and the inference phase, where this model is used to perform the actual inference.

Overview:
- [Flight Delay Example](Flight-Delay-Example)
- [Parameters](Parameters)
-[Package Import](Package-Import)
- [Data Preparation](Data-Preparation)
- [Model Training](Model-Training)
- [FPGA Model Setup](Model-Setup-for-the-FPGA)
- [FPGA Inference](FPGA-Inference)
- [Comparison](Comparison)




##### Flight Delay Example

We will exercise this example using the [flight-delays](https://www.kaggle.com/usdot/flight-delays/data?select=flights.csv) dataset. The Tree-based machine learning model will try to estimate the delay of a flight depending on 18 features like airports, time, etc.. In case of regression, the delay itself will be estimated; in case of classification, the delay will be binned into dicrete slots representing a range of delay time, and the bin for each sample is estimated. For the getteing started guide, we will stick to regression as it is available for all of sk-learn, XGBoost and LightGBM in the current release.

If you want to see the whole picture, look at the scripts in the [scripts](../scripts/)  folder.

##### Parameters

We will set some parameters for this example:
```python

    max_number_of_trees = 100
    numTestSamples = 100
    nLoops = 1000
    max_depth= 8
    num_leaves=900

```

These parameters do not yield optimal results on the estimation, but rather serve as a first guess and for demonstration purposes.


##### Package Import

Start by importing the python library of the Inference Engine:
```python

    import XlPluginRandomForest as xl
```

Additionally, import the packages required for data handling and time measurings:

```python 

    import pandas as pd, numpy as np
    import time

```

Depending on the framework, you need to import of course the framework packages. See the scripts for more details.


##### Data Preparation

We will load the dataset (using ```pandas```) and sample an amount of data from it. The data will be split into a training and test set. Ensure that your ```dataset_name``` points to the file ```flight-delays/flights.csv``` which you downloaded.

```python
    data_origin = pd.read_csv(dataset_name)
    data_origin = data_origin.sample(frac = 0.1, random_state=10)


    feature_names = ["MONTH","DAY","DAY_OF_WEEK","AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT",
                     "ORIGIN_AIRPORT","AIR_TIME", "DEPARTURE_TIME","DISTANCE","ARRIVAL_DELAY"]

    data_origin = data_origin[feature_names]
    data_origin.dropna(inplace=True)

    feature_names.remove("ARRIVAL_DELAY") # arrival delay is not a feature but the target
    numFeatures = len(feature_names)

    data = data_origin.copy()
    cols = ["AIRLINE","FLIGHT_NUMBER","DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
    for item in cols:
        data[item] = data[item].astype("category").cat.codes +1

    data_label = data["ARRIVAL_DELAY"]
    data = data.drop(["ARRIVAL_DELAY"], axis=1)

    print(data.head(5))

    x_train, x_test, y_train, y_test = train_test_split(data, data_label,random_state=10, test_size=0.25)

    x_test = x_test.head(numTestSamples)
    y_test = y_test.head(numTestSamples)
```

As you can see, we loaded and cleaned the data set, replaced categorical features by numerical values, and performed a split. 

##### Model Training

For training, we use these simple parameters for the given framework. They do not necessarily yield good predictions, but serve as demonstration.

- sk-learn:
    ```python
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(max_depth=max_depth,n_estimators=max_number_of_trees, n_jobs=32, random_state=1234)
        fit(x_train, y_train)
    ```
- XGBoost:
    ```python
        import xgboost as xgb
        from xgboost import DMatrix as DMatrix
        model = xgb.XGBRegressor(objective="reg:squarederror", booster="gbtree", random_state=42, max_depth=max_depth, n_estimators=max_number_of_trees, base_score=0.0, n_jobs = 32)
        model.fit(x_train,y_train)
    ```
- LightGBM:
    ```python
       import lightgbm as lgb
       model_regression = lgb.LGBMRegressor(learning_rate=0.15, num_leaves=num_leaves, max_depth=max_depth, n_estimators=max_number_of_trees)
       model_regression.fit(x_train,y_train)
    ```

With these models, we can continue to transform them into FPGA-suitable models, and perform a pure-CPU-base inference to comapre the results later.


##### Model Setup for the FPGA Engine

Now we have a machine learning model which can be used to perform an inference on a CPU. In order to perform the inference using the Tree Inference Engine, we need to create a version of this model that can be loaded into the FPGA. The reason to prepare this model is to save time every time the inference is requested. The model which we are creating for the FPGA is a python tuple which is composed of:

- an instance of the class XlRfEncodedModel, which is tailored to the FPGA you are using and not modifiable
- a number of classes (int)
- a normalization value (float)
- the number of trees in the model

This tuple is put together by a 'setup' class which is framework-dependent. To get an instance of a setup class, use the following code:

- sk-learn:

     ```python
    setup = xl.XlRandomForestSetup() 
    ```

- XGBoost:

    ```python
    setup = xl.XlXGBSetup()
    ```

- LightGBM:

    ```python
    setup = xl.XlLightGBMSetup()
    ```

**Note:** The setup class is the same for regression, binomial and multinomial estimators.

Next, we want to retrieve the model tuple from the created ```setup``` instance. This is achieved using the ```getModelForFPGA``` method. The parameters differ depending on the framework:

- sk-learn:

    ```python
    model_fpga = setup.getModelForFPGA(model)
    ```

    sk-learn provides all required information in the RandomForestRegressor (RandomForestClassifier resp.) object, so that it is the only parameter.

- XGBoost:

    ```python
    model_fpga = setup.getModelForFPGA(model.get_booster(), max_depth, num_classes, feature_names)
    ```

    XGBoost does not provide all information required for the FPGA engine in the booster, so we need to pass the maximum depth of trees, number of classes and feature names (list) additionally.
    In case of regression, set num_classes to 1.

- LightGBM:

    ```python
    json_dump = json.dumps(model.booster_.dump_model())
    model_fpga = setup.getModelForFPGA(json_dump, max_depth)
    ```

    The LightGBM setup class requires a json dump of the LightGBM model and the maximum tree depth.


**Note:** The tuple can be stored to a file using ```pickle```, so that setup and inference phase can be separated.

**Note:** In the current release, you must run the setup of the FPGA model on a system which access to the same FPGA (and only that type) as you will perform the inference on. This is required because the encoded model is created according to the used device.

##### FPGA Inference

Next is the actual inference. This is equal for all three frameworks. First, we create an instance of the inference engine and load the model:

```python

    engine = xl.XlRfInference()
    engine.setModel(model_fpga)

```

Our samples to infer are a ```numpy.ndarray``` with the shape ```(numSamples, numFeatures)```, the data type ```numpy.float32```, and in column-major order (```order='C'```):
```python
    samples = np.ndarray(x_test, dtype=np.float32, order='C')
```

 Additionally, we measure the time over a number of iterations to get an average:

```python
    time_total_fpga = 0
    for n in range(nLoops):
        start_time = time.perf_counter()

        # actual prediction call:
        predictions_fpga = engine.predict(samples)

        end_time = time.perf_counter()
        time_total_fpga = += (end_time - start_time)
    time_total_fpga /= nLoops

```



##### Comparison

Now we might compare the FPGA inference against CPU-based inference in terms of results and runtime. First, we do the inference in software with the 3 frameworks. The syntax is the same as all of them
offer a sk-learn style interface.


```python

    time_total_sw = 0
    for n in range(nLoops):
        start_time = time.perf_counter()
        predictions_sw = model.predict(samples)
        end_time = time.perf_counter()
        time_total_sw += (end_time - start_time)
    time_total_sw /= nLoops
```

When all inferences are done on the FPGA, we can safely shutdown the engine and release the FPGA:

```python
    del engine
    xl.shutdown()
```


Finally, we compute the error on both the inferred data by the CPU and the FPGA, and print the time spent. ```python

```python
    error_sw = abs(y_test - predictions_sw)
    error_fpga = abs(y_test - predictions_fpga) 

    print("Error SW: ", error_sw.mean())
    print("Error FPGA: ", error_fpga.mean())
    print("SW time: ", time_total_sw, "s (average over ", nLoops, " iterations")
    print("FPGA time: ", time_total_fpga, "s (average over ", nLoops, " iterations")
    
```

**Note:** The first inference done with an instance of the engine typically takes more time than subsequent requests since the FPGA needs to be set up initially. For optimal performance in the current release, try to use the same amount of samples in subsequent requests.

For a complete overview, take a look at the [scripts](../scripts/) and use the builtin ```help``` function to explore the python library.
