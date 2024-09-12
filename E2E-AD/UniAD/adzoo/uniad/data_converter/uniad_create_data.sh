
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./create_data.py nuscenes --root-path ./adzoo/uniad/data/nuscenes/ \
       --out-dir ./adzoo/uniad/data/infos/ \
       --extra-tag nuscenes \
       --version v1.0 \  # if you want to use v1.0-mini, you can change it to v1.0-mini
       --canbus ./adzoo/uniad/data/nuscenes/ \
