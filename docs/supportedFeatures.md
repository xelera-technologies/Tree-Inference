
Since the Xelera decision Tree Inference is suited only for inference of tree-based models. Not all features of the supported frameworks are reflected in it. Below you can find general usage restrictions and current framework-specific limitations.

##### General limitations

| Parameter/Feature | Limitation    | Notes |
| ----------------- | ----------    | ----- |
| tree depth        | none          | as long as nodes per tree is satisfied |
| number of nodes   | 1024 per tree | ----- |
| number of features| 1024          | ----- |
| type of feature   | numerical only| use numpy functions to encode categorical features |
| number of samples | none          | ----- |
| samples input     | `numpy.ndarray` | use `order=`\`C\`, `dtype=np.float32, shape=(nSamples, nFeatures)` |
| multiprocess      | not supported | in development |

##### scikit-learn

| Parameter/Feature | Limitation    | Notes |
| ----------------- | ----------    | ----- |
| regression        | supported     | ----- |
| binomial classification | supported | ----- |
| multinomial classification | maximum 128 classes | ----- |
| `apply` method (leaf indices)       | not supported | ----- |
| `decision_path` method              | not supported | ----- |
| `predict` method                    | supported     | ----- |
| `predict_proba` method              | supported     | ----- |
| `predict_log_proba` method          | not supported | ----- |
| `score` method                      | not supported | ----- |

For training and all methods other than the prediction (e.g. information about the model, etc), you can use the trained model itself.

##### XGBoost

| Parameter/Feature | Limitation    | Notes |
| ----------------- | ----------    | ----- |
| regression        | supported     | ----- |
| binomial classification | supported | ----- |
| multinomial classification | supported| max 4 classes on Alveo U50 and U200|
| booster | `gbtree` and `dart` only | `gblinear` not supported |
| `predict`:`ntree` | only default value `0` available |----- |
| `predict`:`output_margin | not supported | defaults to `false` |
| `predict`:`pred_leaf` | not supported | defaults to `false` |
| `predict`:`pred_contribs` | not supported | defaults to `false` |
| `predict`:`approx_contribs` | not supported | defaults to `false` |
| `predict`:`pred_interactions` | not supported | defaults to `false` |
| `predict`:`validate_features` | not supported | defaults to `false` |
| `predict`:`training` | not supported | defaults to `false` |
| missing values | not supported | replace missing values with `np.finfo.min()` |

For training and all methods other than the prediction (e.g. information about the model, etc), you can use the trained model itself.

##### LightGBM

| Parameter/Feature | Limitation    | Notes |
| ----------------- | ----------    | ----- |
| regression        | supported     | ----- |
| binomial classification | supported | ----- |
| multinomial classification | supported| max 4 classes on Alveo U50 and U200|
| boosting_type | `gbdt` and `dart` | `rf` in development; `goss` not supported |
| `predict`:`num_iteration` | only default value | default = `None` |
| `predict`:`raw_score` | not supported | defaults to  `false` |
| `predict`:`pred_leaf` | not supported | defaults to  `false` |
| `predict`:`pred_contrib` | not supported | defaults to`false` |
| `predict`:`data_has_header` | not supported | defaults to `false`; input data need to be `np.ndarray` |
| `predict`:`ls_reshape` | not supported | defaults to `false` |
| missing values | not supported | replace missing values with `np.finfo.min()` |


For training and all methods other than the prediction (e.g. information about the model, etc), you can use the trained model itself.
