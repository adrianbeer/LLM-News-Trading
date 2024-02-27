#!/bin/sh  
while true  
do
	wandb sync --sync-all
	sleep 300
done
