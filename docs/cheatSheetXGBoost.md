# XGBoost (XGBoost) Cheat Sheet

The Python library documentation is available as a built-in ```help``` function. As example, to get the help information about `XlRFInference` function, from inside a *python3* shell run:

```
import XlPluginDecisionTreeInference as xl
help(xl.XlRFInference)
```

### Prepare trained model for FPGA

```
xlSetup = xl.XlXGBSetup()
xlSetup.getModelForFPGA(model.get_booster(), max_depth, num_classes, feature_names,"myModelXGB.xlmodel")
del xlSetup
```
where:
* `model` is the trained XGBoost booster which shall be inferred
* `num_classes` is the number of classes which shall be inferred. 1 in case of regression, 2 in case of binomial classification, n>2 in case of multi-class classification
* `max_depth` is the maximum depth of the trees in booster
* `feature_names` is the list of feature names
* `myModelXGB.xlmodel` is the file name of the returned saved model (`.xlmodel` format) for FPGA


### Run inference (prediction) on FPFA

Create an FPGA inference engine:
```
inference_engine = xl.XlRFInference()
```

Load the trained model into FPGA
```
inference_engine.setModel("myModelXGB.xlmodel")
```

Run the prediction on FPGA (non-blocking python call)
```
inference_engine.predict(X_test)
```
where `X_test` is a `numpy.ndarray` of the input samples.

Get the prediction from FPGA (blocking python call)
```
predictions = inference_engine.get_results()
```

Shutdown
```
del inference_engine
xl.shutdown()
```