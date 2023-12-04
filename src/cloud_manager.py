from dask_kubernetes.operator import KubeCluster, make_cluster_spec
import subprocess
import os
import io, tarfile
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


def fn_to_targz_string(fn):
    with io.BytesIO() as bt:
        with tarfile.open(fileobj=bt,mode='w:gz') as tf:
            tf.add(fn,arcname=os.path.basename(fn))
        bt.seek(0)
        s=bt.read()
    return s


def spin_up_cluster():
    # Google Cloud Authentification
    # this is how the OAuth2 token is available on your cluster
    # token = os.environ.get("CLOUDSDK_AUTH_ACCESS_TOKEN")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/extreme-lore-398917-ac46de419eb2.json"

    cluster: KubeCluster = KubeCluster(custom_cluster_spec="cluster.yaml")
    # Always specify requested memory and cpu to be a little bit less...
    # Good for scheduling and also because googles 4cpu is like 3.92 in reality (sub 4).
    
    # worker_group_config = {
    #     "name": "highmem",
    #     "image": "hermelinkluntjes/thesis:test",
    #     "n_workers": 0,
    #     "resources":{
    #         "requests": 
    #             {"memory": "20Gi",
    #             "cpu": "20"}
    #     },
    #     "args": ["dask-worker", "--no-dashboard" , '--name', "$(DASK_WORKER_NAME)", '--nprocs', '16', '--nthreads', '2']
    # }
    # cluster.add_worker_group(**worker_group_config)
    
    return cluster#, token
   