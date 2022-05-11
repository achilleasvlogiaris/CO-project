# Correlation Optimization

Framework based on: ElectroMagnetic Mining Array (EMMA) (https://github.com/rpp0/emma)

## Redis Installation on Ubuntu

Follow the 2 steps to install and setup Redis: https://www.digitalocean.com/community/tutorials/how-to-install-and-secure-redis-on-ubuntu-20-04


## Installation of the virtual environment

The recommended way is via `venv`:
```bash
$ cd <emma_root_directory>
$ python -m venv env
$ source env/bin/activate
$ pip install -r my_requirements.txt
$ emma.py -h
```


## Configuration

Two config files should be added to the EMMA root directory: `settings.conf` and `datasets.conf`.


### `settings.conf`
#### Example configuration

```
[Network]
broker = redis://:password@redisserver:6379/0
backend = redis://:password@redisserver:6379/0

[Datasets]
datasets_path = /home/user/my-dataset-directory/
stream_interface = eth0

[EMMA]
remote = True
```


### `datasets.conf`
#### Example configuration
dataset directory path should follow this format: /home/ASCAD/ASCAD_data/ASCAD_databases/ASCAD.h5
```
# If the dataset is the ASCAD.h5 file then configure as bellow: 
[ASCAD]
format=ascad
reference_index=0
```
