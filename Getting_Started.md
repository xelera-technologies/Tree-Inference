## Getting Started

In this document, you will be guided through an example which shows you how to leverage the Xelera Tree Inference Engine. It will cover the setup phase, in which a trained model from one of the supported frameworks is transformed into a model that can be loaded to the FPGA, and the inference phase, where this model is used to perform the actual inference.

Contents:
- [Preconditions](#Preconditions)
- [Flight Delay Example](Flight-Delay-Example)
- [Data Preparation](Data-Preparation)
- [Model Training](Model-Training)
- [FPGA Model Setup](Model-Setup-for-the-FPGA)
- [FPGA Inference](FPGA-Inference)
- [Comparison](Comparison)




### Preconditions

Make sure you are running on a system where XeleraSuite, the Tree Inference Plugin and the python library are installed. This is given if you are running on a docker container shipped by Xelera or on the AWS AMI. If you have question, contact support@xelera.io.

### Flight Delay Example

We will exercise this example using the flight-delays TODO link dataset. The Tree-based machine learning model will try to estimate the delay of a flight depending on 18 features like airports, time, etc.. In case of regression, the delay itself will be estimated; in case of classification, the delay will be binned into dicrete slots representing a range of delay time, and the bin for each sample is estimated. For the getteing started guide, we will stick to regression as it is available for all of sk-learn, XGBoost and LightGBM in the current release.

If you want to see the whole picture, look at the scripts in the scripts (TODO LINK) folder.




### Data Preparation

### Model Training

### Model Setup for the FPGA Engine

Now we have a machine learning model which can be used to perform an inference on a CPU. In order to perform the inference using the Tree Inference Engine, we need to create a version of this model that can be loaded into the FPGA. The reason to prepare this model is to save time every time the inference is requested. The model which we are creating for the FPGA is a python tuple which is composed of:

- an instance of the class XlRfEncodedModel, which is tailored to the FPGA you are using and not modifiable
- a number of classes (int)
- a normalization value (float)
- the number of trees in the model

This tuple is put together by a 'setup' class which is framework-dependent. To get an instance, use the following code of a setup class, use the following code:

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

Next, we want to retrieve the model tuple from the created ```setup``` instance. This is achieved using the ```GetModelForFPGA``` method. The parameters differ depending on the framework:

- sk-learn:

    ```python
    model_fpga = setup.GetModelForFPGA(model)
    ```

    sk-learn provides all required information in the RandomForestRegressor (RandomForestClassifier resp.) object, so that it is the only parameter.

- XGBoost:

    ```python
    model_fpga = setup.GetModelForFPGA(model.get_booster(), max_depth, num_classes, feature_names)
    ```

    XGBoost does not provide all information required for the FPGA engine in the booster, so we need to pass the maximum depth of trees, number of classes and feature names (list) additionally.
    In case of regression, set num_classes to 1.

- LightGBM:

    ```python
    json_dump = json.dumps(model.booster_.dump_model())
    model_fpga = setup.GetModelForFPGA(json_dump, max_depth)
    ```

    The LightGBM setup class requires a json dump of the LightGBM model and the maximum tree depth.


**Note:** The tuple can be stored to a file using ```pickle```, so that setup and inference phase can be separated.

**Note:** In the current release, you must run the setup of the FPGA model on a system which access to the same FPGA (and only that type) as you will perform the inference on. This is required because the encoded model is created according to the used device.

### FPGA Inference

Now, we want to use the generated model



### Comparison

Now we might compare the FPGA inference against CPU-based inference in terms of results and runtime. Therefore, we perform a run 

**Note:** The first inference done with an instance of the engine typically takes more time than subsequent requests since the FPGA needs to be set up initially. For optimal performance in the current release, try to use the same amount of samples in subsequent requests