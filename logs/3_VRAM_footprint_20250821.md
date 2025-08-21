# VRAM footprint analysis  
- **GPU**: NVIDIA GeForce RTX 2050 (4 GB VRAM)  
  - Driver: 575.64.05  
  - CUDA Runtime: 12.9  

##  Proof
```
(tu9-mpp) kasif-ak@fedora:~/Projects/VOC-Compliance-Checker$ nvidia-smi
Thu Aug 21 13:44:39 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.64.05              Driver Version: 575.64.05      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2050        Off |   00000000:01:00.0 Off |                  N/A |
| N/A   55C    P8              4W /   30W |      11MiB /   4096MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2806      G   /usr/bin/gnome-shell                      1MiB |
+-----------------------------------------------------------------------------------------+

```
---