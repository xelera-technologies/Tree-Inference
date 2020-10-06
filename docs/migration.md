# Migration from version 0.3.0b3 to 0.4.0b4

If you have previously used version 0.3.0b3 of the Tree Inference Engine, you may have need to update to the changed python API. This page lists the differences.

## Module Import

The module name has been updated.

### Description of changes

The module name has been harmonized to 'XlPluginDecisionTreeInference'.

### Previously (v0.3.0b3)

```import XlPluginRandomForest```

### Update (v0.4.0b4)

```import XlPluginDecisionTreeInference```



## FPGA Model setup

How to setup a fpga-inferable model ('XlModel') from an existing scikit-learn/XGBoost/LightGBM model.

### Description of changes

Transcoding a model to a XlModel can now be done without an Alveo card ('offline').
The setup function can either return the XlModel as python object (which is pickle-able), or store it as file under a given filename. In the latter case, the filename is returned.

### Previously (v0.3.0b3)
|            Framework        |     Call                   |
| :-------------------------: |:-------------------------: |
|  scikit-learn | ```import XlPluginRandomForest as xl; xlSetup = xl.XlRandomForestSetup(); fpga_model = xlrfsetup.getModelForFPGA(clf)``` (1) |  
|  XGBoost      | ```import XlPluginRandomForest as xl; xlSetup = xl.XlXGBSetup(); fpga_model = xlrfsetup.getModelForFPGA(model.get_booster(),  max_depth, num_classes, feature_names)``` (1) | 
|  LightGBM     | ```import XlPluginRandomForest as xl; xlSetup = xl.XlLightGBMSetup(); fpga_model = xlrfsetup.getModelForFPGA(jsonDump, max_depth)``` (1) | 

(1) Alveo card must be available at time of call; else, fails.

### Update (v0.4.0b4)

To retrieve the XlModel as python object, use:

|            Framework        |     Call                   |
| :-------------------------: |:-------------------------: |
|  scikit-learn | ```import XlPluginDecisionTreeInference as xl; xlSetup = xl.XlRandomForestSetup(); fpga_model = xlrfsetup.getModelForFPGA(clf)``` (1, 2) |  
|  XGBoost      | ```import XlPluginDecisionTreeInference as xl; xlSetup = xl.XlXGBSetup(); fpga_model = xlrfsetup.getModelForFPGA(model.get_booster(),  max_depth, num_classes, feature_names)``` (1, 2) | 
|  LightGBM     | ```import XlPluginDecisionTreeInference as xl; xlSetup = xl.XlLightGBMSetup(); fpga_model = xlrfsetup.getModelForFPGA(jsonDump, max_depth)``` (1,2) | 

(1) no FPGA required at time of call
(2) returns XlModel as python object

To store the transcoded XlModel as file:

|            Framework        |     Call                   |
| :-------------------------: |:-------------------------: |
|  scikit-learn | ```import XlPluginDecisionTreeInference as xl; xlSetup = xl.XlRandomForestSetup(); fpga_model = xlrfsetup.getModelForFPGA(clf, 'path/name.xlmodel')``` (1, 3) |  
|  XGBoost      | ```import XlPluginDecisionTreeInference as xl; xlSetup = xl.XlXGBSetup(); fpga_model = xlrfsetup.getModelForFPGA(model.get_booster(),  max_depth, num_classes, feature_names, 'path/name.xlmodel')``` (1, 3) | 
|  LightGBM     | ```import XlPluginDecisionTreeInference as xl; xlSetup = xl.XlLightGBMSetup(); fpga_model = xlrfsetup.getModelForFPGA(jsonDump, max_depth, 'path/name.xlmodel')``` (1,3) | 

(1) no FPGA required at time of call
(2) stores model to ```path/filename.xlmodel``` and returns the resulting path as string. If the file extension ```.xlmodel``` is not given, an exception is thrown.


## FPGA Model loading

Since the way XlModels are stored has change, the way to load them has also changed.

### Description of changes

A XlModel can now be loaded from a file path or given as python object.

### Previously (v0.3.0b3)

|            Framework        |     Call                   |
| :-------------------------: |:-------------------------: |
|  all | ```inference_engine = xl.XlRFInference(); inference_engine.setModel(fpga_model)``` (1) |  

(1) ```fpga_model``` is a python object

### Update (v0.4.0b4)

|            Framework        |     Call                   |
| :-------------------------: |:-------------------------: |
|  all | ```inference_engine = xl.XlRFInference(); inference_engine.setModel(fpga_model)``` (1) |  
|  all | ```inference_engine = xl.XlRFInference(); inference_engine.setModel('path/filename.xlmodel)``` (2) |  

(1) ```fpga_model``` is a python object
(2) ```'path/filename.xlmodel'``` points to the file from the setup phase.


## Inference

There are now 2 modes to call the inference: non-blocking and blocking.

### Description of changes

Blocking mode immediately returns the results for one predict call, which is latency optimized.
Non-blocking mode allows to place concurrent requests and asynchronously fetch the result. This allows for a higher throughput.

### Previously (v0.3.0b3)
|            Framework        |     Call                   |
| :-------------------------: |:-------------------------: |
|  all | ```result = inference_engine.predict(samples)``` (1) |  

(1) Blocks until call is finished; then, ```result``` contains valid data

### Update (v0.4.0b4)

Blocking mode:
|            Framework        |     Call                   |
| :-------------------------: |:-------------------------: |
|  all | ```result = inference_engine.predict_blocking(samples)``` (1) |  

(1) Blocks until call is finished; then, ```result``` contains valid data

Non-Blocking mode:
|            Framework        |     Call                   |
| :-------------------------: |:-------------------------: |
|  all | ```inference_engine.predict(samples); result = engine.get_results()``` (2) |  

(2) ```inference_engine.predict(samples)``` returns immediately.

To test the higher throughput, you can use the non-blocking mode like this:

```
        nLoops = 1000
        hw_start_time = time.perf_counter()
        for i in range(nLoops):
            inference_engine.predict(samples)
        for i in range(nLoops):
            y_pred = inference_engine.get_results()
        hw_stop_time = time.perf_counter()
        hw_time = (hw_stop_time - hw_start_time)/nLoops
```






