#!/bin/bash
set -e
sudo yum -y install git
git clone https://github.com/commoncrawl/cc-pyspark
cd ./cc-pyspark
yes | python3 -m pip install -r requirements.txt