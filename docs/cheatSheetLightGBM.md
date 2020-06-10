# LightGBM (LightGBM) Cheat Sheet


### Prepare trained model for FPGA

```
xlSetup = xl.XlLightGBMSetup()
bdump = model.booster_.dump_model()
jsonModelDump = json.dumps(bdump)
fpga_model_regression = xlSetup.getModelForFPGA(jsonModelDump, max_depth)
```
where:
* `model` is the trained model with LighGBM
* `max_depth` is the max depth of all the trees in the forest

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
