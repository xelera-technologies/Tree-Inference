# Random Forest (scikit-learn) Cheat Sheet

The Python library documentation is available as a built-in ```help``` function. As example, to get the help information about `XlRFInference` function, from inside a *python3* shell run:

```
import XlPluginRandomForest as xl
help(xl.XlRFInference)
```


### Prepare trained model for FPGA

```
xlSetup = xl.XlRandomForestSetup()
xlSetup.getModelForFPGA(model,"myModelRF.xlmodel")
del xlSetup
```
where:
* `model` is a trained scikit-learn Random Forest model which shall be inferred. 
* `myModelRF.xlmodel` is the file name of the returned saved model (`.xlmodel` format) for FPGA

### Run inference (prediction) on FPFA

Create an FPGA inference engine:
```
inference_engine = xl.XlRFInference()
```

Load the trained model into FPGA
```
inference_engine.setModel("myModelRF.xlmodel")
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
