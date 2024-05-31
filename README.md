README
================
Hardware set up: Intel Xeon Platinum 8369B CPU @270GHz, 512GB memory

### 1. Prepare DOCKER envrionment to build and test the ACE compiler
Refer to the Dockerfile under the artifact directory.
```
cd ${artifact_dir}
docker build -t ace:latest .
docker run -it --name ace --privileged ace:latest bash
```

### 2. Build ACE compiler
Under the container directory "/app", run:
```
/app/scripts/build-cmplr.sh Release
```
Then the ACE compler would be build under "/app/release" and install under dir "/app/ace_cmplr".

### 3. Run performance tests for compilation time, run time and runtime memory consumption
Note: given hardward setups, it would take around 5 hours to complete ACE only tests in single thread, and it would take around 13 hours to complete EXPERT only tests in single thread. Running EXPERT tests (-a or -e options below) require more than 300GB memory, it is recommanded to run on machine with at least 400GB memory.

#### 3.1 Build EXPERT program
Under the container directory "/app", run:
```
python3 /app/FHE-MP-CNN/build_cnn.py
```
Expert source will be pulled to "/app/FHE-MP-CNN/FHE-MP-CNN", and executables will be built under "/app/FHE-MP-CNN/FHE-MP-CNN/cnn_ckks/build_cnn".


#### 3.2 Run all ACE and EXPERT performance test in on command
Under the container directory "/app", run:
```
python3 /app/scripts/perf.py -a
```
A log named with the date the command line is launched is generated, for example '2024_05_26_13_18.log'. You can refer to the log for the performance data or the failure info. For example, if you see a "failed due to SIGKILL", it's quite possible that your are run out of memory for an EXPERT case. If you succeed, run:
```
python3 /app/scripts/generate_figures.py -f 2024_05_26_13_18.log
```
It will generate performance figures in paper as 'comp_time.pdf', 'exec_time.pdf' and 'exec_mem.pdf'.
Also the performance comparison results will be displayed on the screen like:
```
ACE speed up: 2.24, Conv reduced: 31.5%, Bootstrap reduced: 63.3%, ReLU reduced: 44.6%
TotalMemory reduced: 84.8% KeyMemory reduced: 87.1%
```
Also you can refer to *.log for the raw data.

#### 3.3 Run ACE and EXPERT separately
Under the container directory "/app", for ACE run:
```
python3 /app/scripts/perf.py
```
It will produce a [date_ace_run].log, and for EXPERT run:
```
python3 /app/scripts/perf.py -e
```
It will produce [date_exp_run].log, to generate comparison figures, run:
```
python3 /app/scripts/generate_figures.py -af [date_ace_run].log -ef [date_exp_run].log
```

#### 3.4 Run ACE only
We placed the data we used in the paper as 'ace_pre.log' and 'expert_pre.log', you can reproduce our paper figures by simply running
```
python3 /app/scripts/generate_figures.py
```
Under the container home dir.
Or you can use our expert data to generate comparison figures for your own ace run:
```
python3 /app/scripts/generate_figures.py -af [date_ace_run].log
```

### 4. Test resnet inference accuracy
Even though we've reduced the accuracy test size to 1000, it will still take 5000+ hours (208+ days) to complete, this is impractical. So we have to turn to parallel execution.

#### 4.1 Build ACE with OPENMP support
Under the container directory "/app", run:
```
/app/scripts/build_cmplr_omp.sh Release
```
Then the OPENMP version of ACE compler would be build under "/app/release_openmp" and install under dir "/app/release_openmp/driver".

#### 4.2 Run ACE accuracy test
Under the container directory "/app", run:
```
/app/scripts/accuracy.sh resnet20_cifar10 0 10
```
It will try to complete the inference on images from index 0 to 10 (excluded) in FHE in parallel run with physical cpu cores available in the system. After complete, a log named with the model and image range will be generated, for example resnet20_cifar10.acc.0.9.log. To infer images from index 100 to 110 (excluded), using command "/app/scripts/accuracy.sh resnet20_cifar10 100 110". All supported model names are:
*    resnet20_cifar10
*    resnet32_cifar10
*    resnet32_cifar100
*    resnet44_cifar10
*    resnet56_cifar10
*    resnet110_cifar10

