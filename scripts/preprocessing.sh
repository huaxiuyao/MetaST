#!/usr/bin/env bash
python maml/preprocess.py --filename=/taxi/nyc_10_20_3600.npz --cluster_file=/cluster/cluster_res_norm_nyc --save_filename=nyc_seq.npz