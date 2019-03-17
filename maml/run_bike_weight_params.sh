#!/usr/bin/env bash

python train_model.py --cities=bike_nyc,bike_dc --save_dir=./models --model_type=bike_att_metatrain_w2 --update_batch_size=128 --test_num_updates=5 --threshold=0 --mem_dim=8 --cluster_loss_weight=1e-2 --meta_lr=1e-5 --update_lr=1e-5 --iterations=20000 --gpu_id=7

python train_model.py --cities=bike_nyc,bike_dc --save_dir=./models --model_type=bike_att_metatrain_w3 --update_batch_size=128 --test_num_updates=5 --threshold=0 --mem_dim=8 --cluster_loss_weight=1e-3 --meta_lr=1e-5 --update_lr=1e-5 --iterations=20000 --gpu_id=7

python train_model.py --cities=bike_nyc,bike_dc --save_dir=./models --model_type=bike_att_metatrain_w5 --update_batch_size=128 --test_num_updates=5 --threshold=0 --mem_dim=8 --cluster_loss_weight=1e-5 --meta_lr=1e-5 --update_lr=1e-5 --iterations=20000 --gpu_id=7

python train_model.py --cities=bike_nyc,bike_dc --save_dir=./models --model_type=bike_att_metatrain_w6 --update_batch_size=128 --test_num_updates=5 --threshold=0 --mem_dim=8 --cluster_loss_weight=1e-6 --meta_lr=1e-5 --update_lr=1e-5 --iterations=20000 --gpu_id=7