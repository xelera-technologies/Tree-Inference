# Random Forest (scikit) Cheat Sheet


### Prepare trained model for FPGA

```
xlSetup = xl.XlRandomForestSetup()
fpga_model = xlSetup.getModelForFPGA(model)
del xlSetup
```
where `model` is the trained model with Scikit. A trained model (`fpga_model`) prepared for FPGA is returned.

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
where `X_test` is a `numpy.ndarray` of the input samples. Predictions are returned.

Shutdown
```
del inference_engine
xl.shutdown()
```
