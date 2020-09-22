# Xelera decision Tree Inference

<p align="center">
<img src="docs/images/Tree_Inference_overview.png" align="middle" width="500"/>
</p>

**Xelera decision Tree Inference** provides FPGA-accelerated inference (prediction) for real-time Classification and Regression applications when high-throughput or low-latency matters. It supports **Random Forest**, **XGBoost** and **LightGBM** algorithms. The user should first train its own model using one of the supported frameworks (**scikit-learn**, **XGBoost**, **LightGBM**, **H20.ai** and **H20 Driverless AI**) and then load and run the prediction via a Python call to Xelera decision Tree Inference Library.


Additional resource:
* [Random Forest Inference Benchmark on Lenovo Thinksystem SE350](https://xelera.io/assets/downloads/Benchmarks/benchmark-001---edge-server-random-forest-inference.pdf)
* [Blog post on accelerating decision tree-based predictive analytics](https://xelera.io/blog/acceleration-of-decision-tree-ensembles)

## What's New
[Release notes](docs/releaseNotes.md)
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

## Acceleration Platforms

|            Cards/Platform            |     Shell        |
| :-------------------------: |:-------------------------: |
|   [Xilinx Alveo U200](https://www.xilinx.com/products/boards-and-kits/alveo/u200.html) | xilinx-u200-xdma-201830.2 |  
|   Nimbix nx_u200_202010| xilinx-u200-xdma-201830.2 |  
|   [AWS f1.2xlarge](https://aws.amazon.com/de/ec2/instance-types/f1/)                     | xilinx_aws-vu9p-f1_shell-v04261818_201920_1 | 

## Features and Limitations
For supported features and current limitations, see [supported parameters](docs/supportedFeatures.md).

## Usage

#### Installation

Xelera decision Tree Inference is available:
* [On-premises](docs/on-premises.md)
* [Nimbix](docs/nimbix.md)
* [AWS](docs/aws.md)

#### Get started with examples
* [Predict the flight delay](docs/exampleFlight.md)

## Cheat Sheet

* [Random Forest](docs/cheatSheetRF.md)
* [XGBoost](docs/cheatSheetXGBoost.md)
* [LightGBM](docs/cheatSheetLightGBM.md)

## Contacts

In case of questions, contact [info@xelera.io](mailto:info@xelera.io)
