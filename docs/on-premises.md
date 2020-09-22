# On-premises


1. [Request License](https://xelera.io/product/demo-license-requests) for the Xelera decision Tree Inference Docker Image and download the License Key File `<license_file>.xlicpak`
2. Host Setup
    1. Clone GitHub Repository for Xilinx Base Runtime: `git clone https://github.com/Xilinx/Xilinx_Base_Runtime.git && cd Xilinx_Base_Runtime`
    2. Run the Host Setup Script: `./host_setup.sh -v 2020.1`
3. Install Docker (if not installed yet)
    1. `cd Xilinx_Base_Runtime/utilities`
    2. `./docker_install.sh`
4. Run the Container

```
tagname="on-premise-u200-0.3.0b3"

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
     --name cont-decision-tree-inference-$USER-$timestamp \
     xeleratechnologies/decision-tree-inference:${tagname} \
     /bin/bash .
```

5. Using a new terminal, load the LICENSE file into the container `<license_file>.xlicpak`:
    * Get the running `<containers_id>`: `docker ps | grep "decision-tree-inference"`
    * `docker cp <license_file>.xlicpak <container_id>:/usr/share/xelera/<license_file>.xlicpak`
