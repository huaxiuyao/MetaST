#!/usr/bin/env bash

python test_model.py --city=bike_new_boston --save_dir=./models --output_dir=./outputs --model_type=bike_att_metatrain_mem4 --test_model=model_19900 --test_days=3 --update_batch_size=128 --mem_dim=4 --threshold=0 --meta_lr=1e-5 --update_lr=1e-5 --epochs=30 --gpu_id=7

python test_model.py --city=bike_new_boston --save_dir=./models --output_dir=./outputs --model_type=bike_att_metatrain_mem8 --test_model=model_19900 --test_days=3 --update_batch_size=128 --mem_dim=8 --threshold=0 --meta_lr=1e-5 --update_lr=1e-5 --epochs=30 --gpu_id=7

python test_model.py --city=bike_new_boston --save_dir=./models --output_dir=./outputs --model_type=bike_att_metatrain_mem16 --test_model=model_19900 --test_days=3 --update_batch_size=128 --mem_dim=16 --threshold=0 --meta_lr=1e-5 --update_lr=1e-5 --epochs=30 --gpu_id=7

python test_model.py --city=bike_new_boston --save_dir=./models --output_dir=./outputs --model_type=bike_att_metatrain_mem32 --test_model=model_19900 --test_days=3 --update_batch_size=128 --mem_dim=32 --threshold=0 --meta_lr=1e-5 --update_lr=1e-5 --epochs=30 --gpu_id=7