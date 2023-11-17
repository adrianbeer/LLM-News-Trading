
from dask_kubernetes.operator import KubeCluster
import subprocess


class VMManager():
    
    def __init__(self, cluster: KubeCluster, worker_group_names = ["highmem"]):
        self.cluster = cluster
        self.worker_group_names = worker_group_names
        
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        self.cluster.scale(0, worker_group=self.worker_group_names[0])
        subprocess.run("gcloud container clusters resize coolercluster --node-pool workerpool --num-nodes 0 --zone us-central1-a --quiet" , 
                       shell=True, 
                       check=True)  