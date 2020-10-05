# AWS


0. [Request License](https://xelera.io/product/demo-license-requests) for the Xelera decision Tree Inference Docker Image and download the License Key File `<license-file>.xlicpak`
1. Login to AWS web console and [launch instance wizard](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LaunchInstanceWizard)
2. Choose an Amazon Machine Image (AMI): **FPGA Developer AMI**
3. Choose an Instance Type: **f1.2xlarge**
4. Configure Instance Details as default
5. Add storage as default
6. Add tags
7. Configure Security Group
8. Review and launch
9. Select an existing key pair or create a new key pair then save it as `<pem-file>.pem` file for ssh
10. Launch Instance. An `<aws-instance-ip>` will be assigned to your instance.
10. Copy `<license-file>.xlicpak` to the launched instance: `scp -i -i <pem-file>.pem <license-file>.xlicpak centos@<aws-instance-ip>:/home/centos/<license-file>.xlicpak`
10. SSH to the created instance: `ssh -i <pem-file>.pem centos@<aws-instance-ip>`
11. (ONLY THE FIRST TIME YOU RUN THE INSTANCE) Install and configure docker:
    * `git clone https://github.com/Xilinx/Xilinx_Base_Runtime.git`
    * `sudo Xilinx_Base_Runtime/utilities/docker_install.sh`
    * `sudo usermod -aG docker centos`
    * log-out and log-in again to the instance
12. Check if the FPGA is usable:
    * Source XRT environment `source /opt/xilinx/xrt/setup.sh`
    * Run `xbutil scan`
    * If it's **usable** the following message should appear:  `[0] 0000:00:1d.0 xilinx_aws-vu9p-f1_dynamic_5_0(ID=0xabcd) user(inst=128)`
    * If it's **not usable**, run `sudo systemctl restart mpd` and check again after a few seconds.
13. Update `<license-file>` value and start the Container running the script:

```
tagname="f1.2xlarge-2020.1-0.4.0b4"
licenseFile="<license-file>.xlicpak"

user=`whoami`
timestamp=`date +%Y-%m-%d_%H-%M-%S`


xclmgmt_driver="$(find /dev -name xclmgmt\*)"
docker_devices=""
echo "Found xclmgmt driver(s) at ${xclmgmt_driver}"
for i in ${xclmgmt_driver} ;
do
  docker_devices+="--device=$i "
done

render_driver="$(find /dev/dri -name renderD\*)"
echo "Found render driver(s) at ${render_driver}"
for i in ${render_driver} ;
do
  docker_devices+="--device=$i "
done

source aws-fpga/vitis_setup.sh

docker run \
     -it \
     --rm \
     $docker_devices \
     -e "TERM=xterm-256color" \
     --mount type=bind,source=${PWD}/$licenseFile,target=/opt/xelera/license.xlicpak,readonly \
     -v ${PWD}/xelera_demo:/app/xelera_demo \
     --name cont-decision-tree-inference-$USER-$timestamp \
     xeleratechnologies/decision-tree-inference:${tagname}
```
