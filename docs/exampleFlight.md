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
