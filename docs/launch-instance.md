# Launch Deep Learning Instance (.sh)

Boka Luo

10/18/2020



## Prerequisite

Before start, I assume you have

* configured `aws` in Linux system or emulator

* cloned repo `deeplearner` on branch `devel/aws` 
* installed `jq` on your Mac, Linux, or emulator

If not

* configure `aws` in terminal

  ```bash
  aws configure
  # AWS Access Key ID []:
  # AWS Secret Access Key []:
  # Default region name []:
  # Default output format []:
  ```
  Make sure that the output type is json (default), as `create_spot_instance.sh` won't work if it isn't.


* clone `deeplearner` on branch `devel/aws`

  ```bash
  git clone -b devel/aws https://github.com/agroimpacts/deeplearner.git
  ```

* install `jq`

  * in `Linux system`

    ```bash
    sudo apt install jq
    ```
  * on `Mac`
  
    ```bash
    brew install jq
    ```

  * in `Git Bash` emulator

    * download `jq-win64.exe` from https://github.com/stedolan/jq/releases, then add this line to `.bashrc` in emulator's home directory

      ```bash
      alias jq=/path/to/jq-win64.exe
      ```

  * in `cygwin` emulator 
    * run `setup.exe`
    * use all the default settings until entering the package installation page
    * search for `jq` in the package installation page, and install its latest version 


* If on a Mac, install `coreutils` to have access to `gdate`

  ```bash
  brew install coreutils
  ```

## Launch Instance and Log in

Step 1: Decide on the instance type. Refer to  [GPU Instances](#gpu-instances) if it is a GPU instance; else, refer to [AWS Instance Types](https://aws.amazon.com/ec2/instance-types/)



Step 2: Add this to `line 3` of `deeplearner/aws_tools/create_spot_instance.sh`, if aws cli is configured in Git Bash`, 

```
source ~/.bashrc
```

Note this line might need to be commented out. 


Step 3: Run `create_spot_instance.sh`

```bash
cd /path/to/deeplearner/aws_tools

./create_spot_instance.sh <ami_id> <instance_type> <security_group_id> \
    <key_pair_name> <iam_role> <new_instance_name> <spot_type
#./create_spot_instance.sh ami-0d2bb8808590fbbc1 g4dn.xlarge \
# sg-0a8bbc91697d6a76b some-key-pair some-iam-role AMItestnew one-time
```



Step 4: Give some time for the spot request to initiate, then log in to the instance

a. get the `public_dns_name` by running, replace the `$INAME` with your instance name

```bash
aws ec2 describe-instances \
	--filters 'Name=tag:Name,Values='"$INAME"'' \
	--output text \
	--query 'Reservations[*].Instances[*].PublicDnsName'
```

b. ssh to the instance

```bash
ssh <user_name>@<public_dns_name>
```



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



### Deep Learning AMI for Geospatial Application

* AMI Name: DL-2
* AMI ID: ami-0c360022958667147

