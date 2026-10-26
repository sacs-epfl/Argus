#!/bin/bash

# Setup environment - copy code from NFS to container
NFS_DIR=... # Absolute path to the NFS directory where the code is stored (e.g., /nfs/ARGUS)
ROOT_DIR=$HOME/ARGUS

echo "==> Setting up ARGUS environment"
echo "  Root directory: $ROOT_DIR"
echo "  NFS directory: $NFS_DIR"

# Create root directory if it doesn't exist
mkdir -p $ROOT_DIR

# Copy code from NFS to container
if [ -d "$NFS_DIR" ]; then
    echo "==> Copying code from NFS"
    (
        cd "$NFS_DIR" || exit 1
        tar --exclude="./eval/data" -cf - .
    ) | (
        cd "$ROOT_DIR" || exit 1
        tar -xf -
    )
else
    echo "==> Warning: NFS directory not found at $NFS_DIR"
    exit 1
fi

# Now use the copied code
cd $ROOT_DIR/scripts || { echo "Error: script directory not found"; exit 1; }

# Add the src directory to PYTHONPATH so decentralizepy can be imported
export PYTHONPATH=$ROOT_DIR/src:$PYTHONPATH

decpy_path=../eval # Path to eval folder
graph=topologies/regular_16.txt # Graph file, relative to scripts/ (see topologies/ or generate_graph.py)
run_path=$NFS_DIR/eval/data # Path to the folder where the graph and config file will be copied and the results will be stored
config_file=configuration_files/config_TinyImageNet_BaDFL.ini

# Create the data directory if it doesn't exist
mkdir -p $run_path
mkdir -p $run_path/configuration_files
mkdir -p $run_path/topologies

# Copy config and graph to run_path
cp $graph $run_path/topologies/

# Update config to use NFS dataset location
sed "s|train_dir = ../eval/data/|train_dir = $NFS_DIR/eval/data/|g" $config_file | \
sed "s|test_dir = ../eval/data/|test_dir = $NFS_DIR/eval/data/|g" > $run_path/$config_file

env_python=... # Path to python executable of the environment | conda recommended
machines=1 # number of machines in the runtime
iterations=160
test_after=10
eval_file=$decpy_path/testing.py # decentralized driver code (run on each machine)
log_level=INFO # DEBUG | INFO | WARN | CRITICAL

m=0 # machine id corresponding consistent with ip.json
echo M is $m

procs_per_machine=16
echo procs per machine is $procs_per_machine

log_dir=$run_path/$(date '+%Y-%m-%dT%H:%M')-tinyimagenet-badfl/machine$m # in the eval folder
mkdir -p $log_dir

$env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $run_path/$graph -ta $test_after -cf $run_path/$config_file -ll $log_level -wsd $log_dir