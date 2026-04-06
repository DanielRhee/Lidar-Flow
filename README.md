The AIEA lab @ UCSC for the Lidar Scene Flow 2026 challenge
# Access the Nautilus PVCs by running
```
sh setup.sh 0
```
and terminate with
```
sh setup.sh 1
```

The datasets are stored in an 8TB pvc called 'lidarflow' in the aiea-slugbotics namespace

# Dataset
The datasets can be downloaded with the shell script in ```/downloadDataset```
git cloning and running:
```
bash resumeDownload.sh
```
will download the dataset and resume if there is an unexpected connection loss. The screen command also works with the original script from agroverse https://argoverse.github.io/user-guide/getting_started.html#downloading-the-data
