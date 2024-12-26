#!/bin/zsh
# Script for runnign `isaacsim-behave` for a number of repetitions,
# specified as a single numeric argument

count=$1
source $HOME/.config/env_config/isaac-sim.sh

while [ $count -gt 0 ]
do
    # Print the values
    echo $count
    isaacsim-behave
    # increment the value
    count=`expr $count - 1`
done
