# XGBoost (XGBoost) Cheat Sheet


### Prepare trained model for FPGA

```
xlSetup = xl.XlXGBSetup()
fpga_model = xlSetup.getModelForFPGA(model.get_booster(), max_depth, num_classes, feature_names)
del xlSetup
```
where:
* `model` is the trained model with XGBoost
* `num_classes` is the number of classes of the label
* `max_depth` is the max depth of all the trees in the forest
* `feature_names`.

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
where `X_test` is an `ndarry` of the input Samples. Predictions are returned.

Shutdown
```
del inference_engine
xl.shutdown()
```
