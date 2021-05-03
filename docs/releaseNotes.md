### 0.6.0b6drm (Beta)
* Supported algorithms:
    * Random Forest
        * Regression
        * Classification (Binomial and Multinomial)
    * XGBoost
        * Regression
        * Classification (Binomial and Multinomial on selected Alveo cards)
    * LightGBM
        * Regression
        * Classification (Binomial and Multinomial on selected Alveo cards)
* Supported frameworks:
    * scikit-learn
    * XGBboost
    * LightGBM
* No feature scaling required: float32-based tree traversal algorithm in FPGA
* Kernel optimized for large ensambles and RF classification (greater than hundreds of trees) 
* Single model inference
* Latency optimized inference
* Python interface with non-blocking  calls
* Inference server with a single FPGA card
* DRM-based licensing system

### 0.4.0b4 (Beta)
* Supported algorithms:
    * Random Forest
        * Regression
        * Classification (Binomial and Multinomial)
    * XGBoost
        * Regression
        * Classification (Binomial and Multinomial on selected Alveo cards)
    * LightGBM
        * Regression
        * Classification (Binomial and Multinomial on selected Alveo cards)
* Supported frameworks:
    * scikit-learn
    * XGBboost
    * LightGBM
* Single model inference
* Latency optimized inference
* Python interface with non-blocking  calls
* Inference server with a single FPGA card
* License support

### 0.3.0b3 (Beta)
* Supported algorithms:
    * Random Forest
        * Regression
        * Classification (Binomial and Multinomial)
    * XGBoost
        * Regression
        * Classification (Binomial)
    * LightGBM
        * Regression
        * Classification (Binomial)
* Supported frameworks:
    * scikit-learn
    * XGBboost
    * LightGBM
* Python interface
* Single model inference
* Latency optimized inference (no throughput mode)
