# XGBoost (XGBoost) Cheat Sheet

The Python library documentation is available as a built-in ```help``` function. As example, to get the help information about `XlRFInference` function, from inside a *python3* shell run:

```
import XlPluginRandomForest as xl
help(xl.XlRFInference)
```

### Prepare trained model for FPGA

```
xlSetup = xl.XlXGBSetup()
fpga_model = xlSetup.getModelForFPGA(model.get_booster(), max_depth, num_classes, feature_names)
del xlSetup
```
where:
* `model` is the trained XGBoost booster which shall be inferred
* `num_classes` is the number of classes which shall be inferred. 1 in case of regression, 2 in case of binomial classification, n>2 in case of multi-class classification
* `max_depth` is the maximum depth of the trees in booster
* `feature_names` is the list of feature names

A trained model (`fpga_model`) prepared for FPGA is returned.


### Run inference (prediction) on FPFA

Create an FPGA inference engine:
```
inference_engine = xl.XlRFInference()
```

Load the trained model into FPGA
```
inference_engine.setModel(fpga_model)
```

Run the prediction on FPGA
```
predictions = inference_engine.predict(X_test)
```
where `X_test` is a `numpy.ndarray` of the input Samples. Predictions are returned.

Shutdown
```
del inference_engine
xl.shutdown()
```
