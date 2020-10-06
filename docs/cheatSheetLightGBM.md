# LightGBM (LightGBM) Cheat Sheet

The Python library documentation is available as a built-in ```help``` function. As example, to get the help information about `XlRFInference` function, from inside a *python3* shell run:

```
import XlPluginDecisionTreeInference as xl
help(xl.XlRFInference)
```

### Prepare trained model for FPGA

```
xlSetup = xl.XlLightGBMSetup()
bdump = model.booster_.dump_model()
jsonModelDump = json.dumps(bdump)
fpga_model_regression = xlSetup.getModelForFPGA(jsonModelDump, max_depth, "myModelLightGBM.xlmodel")
```
where:
* `model` is the trained LightGBM booster which shall be inferred
* `max_depth` is the maximum depth of the trees in the trained booster
* `myModelLightGBM.xlmodel` is the file name of the returned saved model (`.xlmodel` format) for FPGA


### Run inference (prediction) on FPFA

Create an FPGA inference engine:
```
inference_engine = xl.XlRFInference()
```

Load the trained model into FPGA
```
inference_engine.setModel("myModelLightGBM.xlmodel")
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
