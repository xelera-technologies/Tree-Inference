# AWS marketplace

1. Launch a new **EC2 f1.2xlarge** Instance using the the [Xelera decision Tree Inference Demo AMI](https://aws.amazon.com/marketplace/pp/B08KG7FNV3?qid=1601476507885&sr=0-1&ref_=srh_res_product_title).
2. SSH to the created instance: `ssh -i <pem-file>.pem centos@<aws-instance-ip>`
3. Start the Container running `./run_xelera_xemo.sh` or the script below:


```
tagname="f1.2xlarge-marketplace-2020.1-0.4.0b4"

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
     -v ${PWD}/xelera_demo:/app/xelera_demo \
     --name cont-decision-tree-inference-$USER-$timestamp \
     xeleratechnologies/decision-tree-inference:${tagname}
```
