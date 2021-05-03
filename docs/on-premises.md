# On-premises


1. Request a license via the [Xilinx App Store](https://appstore.xilinx.com) for the Xelera decision Tree Inference Docker Image and download the credential key File `cred.json`
2. Host Setup
    1. Clone GitHub Repository for Xilinx Base Runtime: `git clone https://github.com/Xilinx/Xilinx_Base_Runtime.git && cd Xilinx_Base_Runtime`
    2. Run the Host Setup Script: `./host_setup.sh -v 2020.2`
3. Install Docker (if not installed yet)
    1. `cd Xilinx_Base_Runtime/utilities`
    2. `./docker_install.sh`
4. Update `<tag-name>` (a list of Docker images is available on [Xelera Technologies Docker Hub](https://hub.docker.com/r/xeleratechnologies/decision-tree-inference) page). The latest tags are:

|            Cards/Platform            |     Tag        |
| :------------------------- |:------------------------- |
|   Xilinx Alveo U50| u50-2020.2-0.6.0b6drm | 
|   Xilinx Alveo U200| u200-2020.2-0.6.0b6drm | 

5. Start the Container running the script:

```
tagname="<tag-name>"
licenseFile="cred.json"

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
     --mount type=bind,source=${PWD}/cred.json,target=/opt/xelera/cred.json,readonly \\
     --name cont-decision-tree-inference-$USER-$timestamp \
     xeleratechnologies/decision-tree-inference:${tagname}
```
