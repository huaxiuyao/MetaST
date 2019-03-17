#!/usr/bin/env bash


python ./maml/train_model.py \
--cities=env_ne,env_pa,env_sw --save_dir=./models --model_type=env_pretrain \
--update_batch_size=128 --test_num_updates=1 --threshold=1e-3 \
--meta_lr=1e-5 --update_lr=1e-5 --iterations=20000 --gpu_id=0

python train_model.py \
--cities=env_ne,env_pa,env_sw --save_dir=./models --model_type=env_metatrain_t5 \
--update_batch_size=128 --test_num_updates=5 --threshold=1e-3 \
--meta_lr=1e-5 --update_lr=1e-5 --iterations=20000 --gpu_id=0

python train_model.py \
--cities=env_ne,env_pa,env_sw --save_dir=./models --model_type=env_att_metatrain_mem8 \
--update_batch_size=128 --test_num_updates=5 --threshold=1e-3 \
--mem_dim=8 --cluster_loss_weight=1e-4 \
--meta_lr=1e-5 --update_lr=1e-5 --iterations=20000 --gpu_id=0

python train_model.py \
--cities=nyc,dc,porto --save_dir=./models --model_type=att_metatrain_mem16 \
--update_batch_size=128 --test_num_updates=5 --threshold=0 \
--mem_dim=16 --cluster_loss_weight=1e-4 \
--meta_lr=1e-5 --update_lr=1e-5 --iterations=20000 --gpu_id=7




python test_model.py --city=env_ne --save_dir=./models --output_dir=./outputs \
--model_type=env_pretrain --test_model=model_19900 --test_days=3 --update_batch_size=128 \
--threshold=1e-3 --meta_lr=1e-5 --update_lr=1e-5 --epochs=30 --gpu_id=0

python test_model.py --city=env_ne --save_dir=./models --output_dir=./outputs \
--model_type=env_metatrain_t5 --test_model=model_19900 --test_days=3 --update_batch_size=128 \
--threshold=1e-3 --meta_lr=1e-5 --update_lr=1e-5 --epochs=30 --gpu_id=2

python test_model.py --city=env_ne --save_dir=./models --output_dir=./outputs \
--model_type=env_att_metatrain_mem4 --test_model=model_19900 --test_days=3 --update_batch_size=128 \
--mem_dim=4 --threshold=1e-3 --meta_lr=1e-5 --update_lr=1e-5 --epochs=30 --gpu_id=2



python test_model.py --city=boston --save_dir=./trained_models/nyc+dc+porto--boston --output_dir=./outputs \
--model_type=nyc+dc+porto_pretrain --test_model=model_19900 --test_days=3 --update_batch_size=128 \
--threshold=0 --meta_lr=1e-5 --update_lr=1e-5 --epochs=30 --gpu_id=0

python test_model.py --city=boston --save_dir=./trained_models/nyc+dc+porto--boston --output_dir=./outputs \
--model_type=metatrain_t5 --test_model=model_19900 --test_days=3 --update_batch_size=128 \
--threshold=0 --meta_lr=1e-5 --update_lr=1e-5 --epochs=30 --gpu_id=0

python test_model.py --city=boston --save_dir=./trained_models/nyc+dc+porto--boston --output_dir=./outputs \
--model_type=att_metatrain_mem8 --test_model=model_19900 --test_days=3 --update_batch_size=128 \
--mem_dim=8 --threshold=0 --meta_lr=1e-5 --update_lr=1e-5 --epochs=10 --gpu_id=3



python train_model.py \
--cities=nyc,dc,porto --save_dir=./models --model_type=att_metatrain_mem8 \
--update_batch_size=128 --test_num_updates=5 --threshold=0 \
--mem_dim=8 --cluster_loss_weight=1e-4 \
--meta_lr=1e-5 --update_lr=1e-5 --iterations=20000 --gpu_id=0


python test_model.py --city=bike_boston --save_dir=./models --output_dir=./outputs \
--model_type=bike_nyc+dc_pretrain --test_model=model_19900 --test_days=3 --update_batch_size=128 \
--threshold=0 --meta_lr=1e-5 --update_lr=1e-5 --epochs=30 --gpu_id=0

python test_model.py --city=bike_boston --save_dir=./models --output_dir=./outputs \
--model_type=bike_metatrain_t5 --test_model=model_19900 --test_days=3 --update_batch_size=128 \
--threshold=0 --meta_lr=1e-5 --update_lr=1e-5 --epochs=30 --gpu_id=0

python test_model.py --city=bike_chicago --save_dir=./models --output_dir=./outputs \
--model_type=bike_att_metatrain_mem8 --test_model=model_19900 --test_days=3 --update_batch_size=128 \
--mem_dim=8  --threshold=0 --meta_lr=1e-5 --update_lr=1e-5 --epochs=10 --gpu_id=7


python test_model.py --city=chicago --save_dir=./models --output_dir=./outputs \
--model_type=att_metatrain_mem8 --test_model=model_19900 --test_days=3 --update_batch_size=128 \
--mem_dim=8 --threshold=0 --meta_lr=1e-5 --update_lr=1e-5 --epochs=20 --gpu_id=0



