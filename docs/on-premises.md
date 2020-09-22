# On-premises


1. [Request License](https://xelera.io/product/demo-license-requests) for the Xelera decision Tree Inference Docker Image and download the License Key File `<license_file>.xlicpak`
2. Host Setup
    1. Clone GitHub Repository for Xilinx Base Runtime: `git clone https://github.com/Xilinx/Xilinx_Base_Runtime.git && cd Xilinx_Base_Runtime`
    2. Run the Host Setup Script: `./host_setup.sh -v 2020.1`
3. Install Docker (if not installed yet)
    1. `cd Xilinx_Base_Runtime/utilities`
    2. `./docker_install.sh`
4. Update `<license_file>` value and start the Container running the script:

```
tagname="u200-2020.1-0.3.0b3"
license_file="<license_file>.xlicpak"

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

docker run \
     -it \
     --rm \
     $docker_devices \
     -e "TERM=xterm-256color" \
     --mount type=bind,source=${PWD}/$license_file,target=/opt/xelera/license.xlicpak,readonly \
     --name cont-decision-tree-inference-$USER-$timestamp \
     xeleratechnologies/decision-tree-inference:${tagname} \
     /bin/bash .
```
