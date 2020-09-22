# AWS


0. [Request access](https://xelera.io/survey-aws-ami-xelera-tree-inference-engine) to the Xelera decision Tree Inference Demo AMI
1. Launch a new EC2 f1.2xlarge Instance using the the Xelera decision Tree Inference Demo AMI. Xelera decision Tree Inference Demo AMI is available under "My AMIs" -> "shared with me" section of the "Step 1: Choose an Amazon Machine Image (AMI)" panel.

<p align="center">
<img src="docs/images/AWS_sharedAMI.png" align="middle" width="500"/>
</p>

2. Connect to the remote EC2 instance. Use `centos` as username.
3. Navigate to the `xelera_demo` folder: `cd /app/xelera_demo`
4. Source the setup script: `source xelera_setup.sh`
