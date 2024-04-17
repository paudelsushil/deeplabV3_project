# Set Up Deep Learning Instance 

Boka Luo & Lyndon Estes

16/2/2022

## Set Up Instance

There are four general stages to set up a full instance environment: 

1. Launch a CPU instance
   * Choose minimal settings, including disk size
2. Stop the instance, change it to a small GPU instance
   * Update NVIDIA drivers
3. Stop GPU instance, change back to CPU instance
   * set up user account
   * install necessary `python` packages
   * configure `jupyter notebook`
4. Launch a GPU instance from the created AMI, then
   * Update NVIDIA drivers
   * test full deep learning codes
5. Save the instance to a new AMI


### Launch Instance (Console)

Step 1: Log in to ec2 dashboard and click `Launch instances` button

Step 2: Choose the most recent Deep Learning AMI running Ubuntu 18.04 (at time Version 56.1, AMI ID `ami-09ac68f361e5f4a13`). Choose an AMI

Step 3: Choose the instance type `t3.xlarge`, then click `Next: Configure Insatance Details`

Step 4: Configure instance details. Note the steps below are for a spot instance, but these can be ignored to create an on-demand instance for configuring. THe main reason for ignoring these is that GPU spot instances are almost never available these days. 
 1. Instance type -- spot instance

    a.  Check `Request Spot Instances` under `Purchasing option` section 

    b. Skip the `maximum price` setting; this will use on-demand price as your bid price

    c. Check `Persistent request` valid for 3 days, and set the `Interruption behavior` as `Stop`

2. Instance zone

   a. find the lowest  price zone by running this in terminal

   ```bash
   ITYPE=t3.xlarge
   # get bid price
   START_TIME=$(date --date="3 days ago" +"%Y-%m-%dT%T")
   END_TIME=$(date +"%Y-%m-%dT%T")
   
   read -r -d '' PRICES << EOF
       $(aws ec2 describe-spot-price-history --instance-types $ITYPE \
   		--product-description Linux/UNIX \
   		--start-time $START_TIME \
   		--end-time $END_TIME)
   EOF
   
   ## find lowest price zone
   ZONE=$(echo $PRICES |\
   	jq '.SpotPriceHistory| sort_by(.AvailabilityZone | explode | map(-.)) |
   	min_by(.SpotPrice | tonumber)|.AvailabilityZone')
   
   
   echo $ZONE
   #us-east-1f
   ```
   
   b. Select `subnet` in the lowest price zone ` us-east-1f`

3. IAM role -- `activemapper_planet_readwrites3`
4. Skip the other settings in the page and click `Next: Add Storage`

Step 4: Add Storage

Set the `Root` size to be 130 GiB, which is the basic requirement by the DL AMI

Step 5: Add Tags `Name` and set the `Value` to be `DL_mapper2`

Step 6: Configure the security group as `labeller-security`, then click ` Review and Launch`

Step 7: Confirm the information are correct and click `Launch` to specify the `key-pair` and launch the instance

### Set up Instance
#### Update NVIDIA drivers
First, stop the instance. When it is stopped, change the instance type to a `g5.xlarge`. Restart the instance. 


##### Log in as root

```
ssh -i <key-pair.pem> ubuntu@<public-ipv4-from-instance>
```

##### Find and install NVIDIA drivers
Following [instructions](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html) to update drivers, choose option 3
for NVIDIA GRID drivers for Ubuntu, follow these steps:

```bash
sudo apt-get update -y
sudo apt-get upgrade -y linux-aws
```

If that doesn't work, because it throws an error for unmet dependencies, then:

```bash
sudo apt-get install aptitude
sudo aptitude install linux-aws
```

Follow the directions until Step 8, which is to download the latest grid drivers. This cannot be achieved from the instance itself because of insufficient IAM permissions, so to achieve this run the command provided:

```bash
aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ .
```

From your local AWS CLI enabled computer, and then:

```bash
scp -i <your-key-pair.pem NVIDIA-Linux-x86_64-470.82.01-grid-aws.run ubuntu@<instance-public-ipv4>:/home/ubuntu/
```
To move the downloaded drivers to the instance.

Continue on until step 10. Run this, but ignore all warnings, and when prompted to install a new module build, select No.

Step after step 12. 

##### Test whether the GPU works
Follow the instructions [here](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-conda.html) for testing `pytorch`, except run `conda activate pytorch_p38` instead. You might have to install `torch` to do this. Use `conda` to do so, although we should note that AWS advises using `pip` to install packages within the `conda` environment. 

##### Exit the instance and stop it

And then switch the instance type back to `t3.xlarge`

#### Install packages
* First activate the necessary conda environment:

    ```bash
    conda activate pytorch_p38
    ```

* Then install `rasterio`:

    ```bash
    conda install -c conda-forge rasterio
    ```

  * test reading in python

    * rasterio

      ```python
      import rasterio
      with rasterio.open('s3://activemapper/planet/composite_sr_buf/GS/tile486215_736815_736967.tif') as src:
          print(src.profile)
      ```

    * gdal (note: this is legacy--gdal is not necessary at this point)

      ```python
      import gdal
      path = '/vsis3/activemapper/planet/composite_sr_buf/GS/tile486215_736815_736967.tif'
      ds = gdal.Open(path)
      ds.ReadAsArray()
      ```
    
* tensorboard for pytorch environment

  ```bash
  conda install tensorboard
  ```

* tensorboardX for pytorch enviroment

  ```bash
  conda install tensorboardX
  conda install soundfile
  ```

#### Set up jupyter notebook

Step 1: Check that python is in the conda environment

```bash
which python
# /home/ubuntu/anaconda3/bin/python
```

Step 2: Create a password for jupyter notebook

```bash
cd ~
ipython

from IPython.lib import passwd
passwd()

# Enter password: 
# Verify password: 
# 'sha1:***********************************************************'
sha1:366919f9495c:c06fabecaa6c15a25a95a5728fe5b3e75efc2943
''

exit
```

Step 3: Create jupyter config profile

```bash
jupyter lab --generate-config
```



Step 4: Create certificates for https

```bash
mkdir certs
cd certs
openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mycert.pem -out mycert.pem
# provide other user the read permission
chmod o+r mycert.pem
```
Follow the prompts to enter the details asked (e.g. country, organization, etc)


Step 5: Configure jupyter 

```bash
cd ~/.jupyter/
vim jupyter_lab_config.py
```

​	Insert this at the beginning of the document, then exit; replace `'sha1:366919f9495c:c06fabecaa6c15a25a95a5728fe5b3e75efc2943'` with the generated key

```bash
c = get_config()

# Kernel config
c.IPKernelApp.pylab = 'inline'  # if you want plotting support always in your notebook

# Notebook config
c.NotebookApp.certfile = u'/home/ubuntu/certs/mycert.pem' #location of your certificate file
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False  #so that the ipython notebook does not opens up a browser by default
c.NotebookApp.password = u'sha1:366919f9495c:c06fabecaa6c15a25a95a5728fe5b3e75efc2943' #the encrypted password we generated above
# Set the port to 8888, the port we set up in the AWS EC2 set-up
c.NotebookApp.port = 8888
```

Step 6: Test jupyter notebook

[comment]: <> (a. Create a folder for notebooks)

[comment]: <> (```bash)

[comment]: <> (cd ~)

[comment]: <> (mkdir Notebooks)

[comment]: <> (cd Notebooks)

[comment]: <> (```)

a. Create new screen

```bash
screen
```

b. Start jupyter notebook, then visit it in browser, by using port`8888`. 

```bash
sudo chown $USER:$USER /home/ubuntu/certs/mycert.pem
jupyter notebook
```

And then changes the permissions on `mycert.pem` to make it usable by another user profile by running the following:

```bash
chmod o+r /home/ubuntu/certs/mycert.pem 
```

c. Visit jupyter notebook in browser, using port 8888. For example, if your ec2 instance has a public IPv4 of `54.226.84.206`, then visit jupyter notebook with `https://54.226.84.206:8888/`

d. Exit jupyter notebook (`ctrl+c`) and detach from screen (`ctrl+a+d`)

* Other useful commands
  * create new window `ctrl+a+c`
  * switch windows `ctrl+a+n`
  * reattach to screen `screen -r`
  * see running screen processes `screen -ls`

#### Set up new user account

Step 1: add new user and log in to the user account

```bash
sudo adduser <new_user> --disabled-password
sudo bash # to gain root access
sudo su - <new_user>
```


Step 2: Add public key and change the file permissions

a. Create an `authorized_keys` file in `.ssh`

```bash
mkdir .ssh
touch .ssh/authorized_keys

chmod 600 .ssh/authorized_keys
```

b. Add public key to `authorized_keys` file

Copy in public keys from computer(s) you want to have access to instance. On local computer, for example, run from terminal/git bash `vim ~/.ssh/id_rsa.pub` (you might have a file with a different name ending in .pub, e.g. one beginning with `id_ed`). Copy the key in there, and then back in the instance, run `vim ~/.ssh/authorized_keys`, and then when in the file type `I` and then ctrl-v to paste in the key, then save and exit by hitting `ESC` followed by `:wq`. 


Step 3: Configure conda and other system packages; insert this into `~/.bashrc` (at the bottom)

```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/ubuntu/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ubuntu/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/ubuntu/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/ubuntu/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# PATHs
# CURL certificates
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
# gdal
export CPLUS_INCLUDE_PATH=/usr/include/gdal  # probably not needed 
export C_INCLUDE_PATH=/usr/include/gdal  # probably not needed
```

Step 4: configure jupyter notebook

a. create jupyter profile for the new user

```bash
jupyter lab --generate-config
```

b. create config file

```bash
cd ~/.jupyter/
vim jupyter_lab_config.py
```

c. insert this to `jupyter_notebook_config.py`, Insert this at the beginning of the document, then exit; replace `'sha1:***********************************************************'` with the generated key from root

```bash
c = get_config()

# Kernel config
c.IPKernelApp.pylab = 'inline'  # if you want plotting support always in your notebook

# Notebook config
c.NotebookApp.certfile = u'/home/ubuntu/certs/mycert.pem' #location of your certificate file
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False  #so that the ipython notebook does not opens up a browser by default
c.NotebookApp.password = u'sha1:***********************************************************' #the encrypted password we generated above
# Set the port to 8888, the port we set up in the AWS EC2 set-up
c.NotebookApp.port = 8888
```
After completing this step, `cd ..`

c. Test jupyter notebook, with command line `jupyter lab` from a separate `screen`. Then Visit jupyter notebook in browser, using port 8888. 

### Clone necessary repositories
First set up a new folder under the <new-user> profile:

```bash
mkdir projects
```

Then:

```bash
cd projects
git clone https://github.com/agroimpacts/deeplearner
```

You may wish to clone other repos for projects, etc. 

--------------------------------------------------------------------------------
Below here needs updating
--------------------------------------------------------------------------------

### Saved configuration

Image on gpu instane

* AMI Name: DLgpu
* AMI ID: ami-0643df3b347740189
* snapshot ID: snap-03f9eee5f214df5f1

 

### Task Finish Note  

| Conda Environment         | Set .err Permission | Install GDAL/rasterio | Install tensorboard | Install tensorboardX | Install tqdm | Install tensorlayer |
| ------------------------- | ------------------- | --------------------- | ------------------- | -------------------- | ------------ | ------------------- |
| base                      | --                  | --                    | --                  | --                   | --           | --                  |
| aws_neuron_mxnet_p36      | --                  | --                    | --                  | --                   | --           | --                  |
| aws_neuron_pytorch_p36    | --                  | √                     | √                   | √                    | --           | --                  |
| aws_neuron_tensorflow_p36 | --                  | √                     | --                  | --                   | √            |                     |
| chainer_p27               | --                  | --                    | --                  |                      | --           | --                  |
| chainer_p36               | --                  | --                    | --                  |                      | --           | --                  |
| mxnet_latest_p37          | --                  | --                    | --                  |                      | --           | --                  |
| mxnet_p27                 | --                  | --                    | --                  |                      | --           | --                  |
| mxnet_p36                 | --                  | --                    | --                  |                      | --           | --                  |
| python2                   | --                  | --                    | --                  |                      | --           | --                  |
| python3                   | --                  | --                    | --                  |                      | --           | --                  |
| pytorch_latest_p36        | √                   | √                     | √                   | √                    | √            | --                  |
| pytorch_p27               | √                   | √                     | √                   | √                    | √            | --                  |
| pytorch_p36               | √                   | √                     | √                   | √                    | √            | --                  |
| tensorflow2_latest_p37    | √                   | √                     | --                  | --                   | √            | √                   |
| tensorflow2_p27           | √                   | √                     | --                  | --                   | √            | --                  |
| tensorflow2_p36           | √                   | √                     | --                  | --                   | √            | √                   |
| tensorflow_p27            | √                   | √                     | --                  | --                   | √            | --                  |
| tensorflow_p36            | √                   | √                     | --                  | --                   | √            |                     |



## Appendix



### GPU Instances

| Instance Series | GPU        |
| --------------- | ---------- |
| P3              | Tesla V100 |
| P2              | NVIDIA K80 |
| G3              | Tesla M60  |
| G4              | NVIDIA T4  |



|       | Instance Type | GPU Count | GPU Memory (GB) | vCPU | Memory (GB) | On-demand Price | Spot Price |
| ----- | ------------- | --------- | --------------- | ---- | ----------- | --------------- | ---------- |
|       | p3.2xlarge    | 1         | 16              | 8    | 61          | 3.060           | 0.918      |
|       | p3.8xlarge    | 4         | 64              | 32   | 244         | 12.240          | 3.672      |
|       | p3.16xlarge   | 8         | 128             | 64   | 488         | 24.480          | 9.4884     |
|       | p3dn.24xlarge | 8         | 256             | 96   | 768         | 31.218          | 10.2938    |
|       | p2.xlarge     | 1         | 12              | 4    | 61          | 0.900           | 0.27       |
|       | p2.8xlarge    | 8         | 96              | 32   | 488         | 7.200           | 2.16       |
|       | p2.16xlarge   | 16        | 192             | 64   | 732         | 14.400          | 4.32       |
| √test | g3s.xlarge    | 1         | 8               | 4    | 30.5        | 0.75            | 0.225      |
|       | g3.4xlarge    | 1         | 8               | 16   | 122         | 1.14            | 0.342      |
| √2    | g3.8xlarge    | 2         | 16              | 32   | 244         | 2.28            | 0.684      |
| √4    | g3.16xlarge   | 4         | 32              | 64   | 488         | 4.56            | 1.368      |
| √test | g4dn.xlarge   | 1         | 16              | 4    | 16          | 0.526           | 0.1578     |
| √1    | g4dn.2xlarge  | 1         | 16              | 8    | 32          | 0.752           | 0.2256     |
| √1    | g4dn.4xlarge  | 1         | 16              | 16   | 64          | 1.204           | 0.3612     |
|       | g4dn.8xlarge  | 1         | 16              | 32   | 128         | 2.176           | 0.746      |
|       | g4dn.16xlarge | 1         | 16              | 64   | 256         | 4.352           | 1.3056     |
| √4    | g4dn.12xlarge | 4         | 64              | 48   | 192         | 3.912           | 1.4146     |
|       | g4dn.metal    | 8         | 128             | 96   | 384         | 7.824           | 2.3472     |



### Tested Python Codes

* Some directories and file name

  ```python
  import os
  import boto3
  # data directory
  dir_home = os.getenv("HOME")
  dir_data = "s3://activemapper/data_DL"
  dir_out = "s3://activemapper/DL/Result/semantic_segmentation/boka/test"
  dir_tmp = os.path.join(dir_home, 'tmp')
  if not os.path.exists(dir_tmp):
  	os.mkdir(dir_tmp)
  # file names
  fn_catalog_train_val = "semantic_catalog_train_val.csv"
  # s3
  bucket = "activemapper"
  prefix_out = "DL/Result/semantic_segmentation/boka/test"
  ```

* rasterio

  ```python
  from datetime import datetime
  import rasterio
  
  fn_img = "tile486234_736815_736967.tif"
  dir_img_s3 = os.path.join(dir_out, fn_img)
  fn_img_new = "tile486234_736815_736967_new.tif"
  dir_img_s3_new = os.path.join(dir_out, fn_img_new)
  
  with rasterio.open(dir_img_s3, "r") as src:
      img =src.read()
      meta = src.meta
   with MemoryFile() as memfile:
          with memfile.open(**meta) as dst:
              dst.write(img)
          s3_client.upload_fileobj(Fileobj = memfile,
                           Bucket = bucket,
                           Key = os.path.join(prefix_out,fn_img_new))
  ```

* csv

  ```python
  # read csv 
  ## make sure aws is configured in the system
  import pandas as pd
  import s3fs
  
  df = pd.read_csv(os.path.join(dir_data, fn_catalog_train_val))
  print(df.head())
  ```

* yaml

  ```python
  # config backup to s3
  import yaml
  dir_config = os.path.join(dir_home, "deeplearner/config.yaml")
  ## read from local
  with open(dir_config, "r") as config:
      params = yaml.safe_load(config)['Train_Validate']
  ## write to tmp file
  fn_config_tmp = "config_train_val_tmp.yaml"
  dir_config_tmp = os.path.join(dir_tmp, fn_config_tmp)
  with open(dir_config_tmp, "w") as dst:
      yaml.dump(params, dst, default_flow_style=False)
  ## upload to s3       
  # use client
  s3_client = boto3.client("s3")
  s3_client.upload_file(Filename = dir_config_tmp, 
                        Bucket = bucket, 
                        Key = os.path.join(prefix_out, fn_config_tmp))
  # delect tmp file
  os.remove(dir_config_tmp)
  ```

* pth

  ```python
  import torch
  
  # load without downloading failed, download first
  fn_params = "unet_params.pth"
  dir_params_tmp = os.path.join(dir_tmp, fn_params)
  # download
  s3_client.download_file(Bucket = bucket,
                         Key = os.path.join(prefix_out, fn_params),
                         Filename = dir_params_tmp)
  
  # params = torch.load(dir_params_tmp)
  params = torch.load(dir_params_tmp,  map_location=torch.device('cpu')) # on a cpu instance
  
  torch.save(params, dir_params_tmp)
  s3_client.upload_file(Filename = dir_params_tmp,
                       Bucket = bucket,
                       Key = os.path.join(prefix_out,"unet_params_new.pth"))
  
  # delete tmp file
  os.remove(dir_params_tmp)
  ```

  
