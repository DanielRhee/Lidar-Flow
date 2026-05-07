The AIEA lab @ UCSC for the Lidar Scene Flow 2026 challenge 

Documentation on function available in docdumentation.md

# Access the Nautilus PVCs by running
```
sh setup.sh 0
```
and terminate with
```
sh setup.sh 1
```

The datasets are stored in an 8TB pvc called 'lidarflow' in the
aiea-slugbotics namespace.

In the PVC run:
```
. start.sh
```
to set up the necessary environments for development/training. It is
*about* as cuda version agnostic as possible, and should train and
install the correct libraries. Training by default is with FP16. BF16
might produce some better results. 

Jobs are available by running 
```
sh setup.sh 3
```
and terminated with 
```
setup.sh 4
```


# Dataset
The datasets can be downloaded with the shell script in the
```/downloadDataset``` directory 

git cloning and running:
```
bash resumeDownload.sh
```
will download the dataset and resume if there is an unexpected
connection loss. The screen command also should work but for some
reason causes pods to frequently fail with exit code 137, so it might
be causing an out of memory error on pods that are low on memory anda
have high output to stdout. 
