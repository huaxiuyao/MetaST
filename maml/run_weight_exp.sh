#!/usr/bin/env bash

python test_model.py --city=chicago --save_dir=./models --output_dir=./outputs --model_type=att_metatrain_w2 --test_model=model_19900 --test_days=3 --update_batch_size=128 --mem_dim=8 --threshold=0 --meta_lr=1e-5 --update_lr=1e-5 --epochs=20 --gpu_id=1
python test_model.py --city=chicago --save_dir=./models --output_dir=./outputs --model_type=att_metatrain_w3 --test_model=model_19900 --test_days=3 --update_batch_size=128 --mem_dim=8 --threshold=0 --meta_lr=1e-5 --update_lr=1e-5 --epochs=20 --gpu_id=1
python test_model.py --city=chicago --save_dir=./models --output_dir=./outputs --model_type=att_metatrain_w5 --test_model=model_19900 --test_days=3 --update_batch_size=128 --mem_dim=8 --threshold=0 --meta_lr=1e-5 --update_lr=1e-5 --epochs=20 --gpu_id=1
python test_model.py --city=chicago --save_dir=./models --output_dir=./outputs --model_type=att_metatrain_w6 --test_model=model_19900 --test_days=3 --update_batch_size=128 --mem_dim=8 --threshold=0 --meta_lr=1e-5 --update_lr=1e-5 --epochs=20 --gpu_id=1
