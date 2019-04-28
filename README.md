# MetaST (Meta-learning for Spatial-Temporal Prediction)

## About
Source code of the paper [Learning from Multiple Cities: A Meta-Learning Approach for Spatial-Temporal Prediction](https://arxiv.org/abs/1901.08518)

If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{yao2019metast,
  title={Learning from Multiple Cities: A Meta-Learning Approach for Spatial-Temporal Prediction},
  author={Yao, Huaxiu and Liu, Yiding and Wei, Ying and Tang, Xianfeng and Li, Zhenhui},
  booktitle={the Web Conference 2019 (WWW'19)},
  year={2019} 
}

```

## Data
- Taxi Data
    - NYC, Washington DC, Porto, Chicago, Boston
    - processed data is in './data/taxi'
- Bike Data
    - NYC, Washington DC, Chicago
    - processed data is in './data/bike'
- Environment PH Data
    - Midwest, Northeast, Pacific, South, Southwest, West
    - processed data is in './data/environment'

## Usage

Please check the data and scripts/preprocessing, training and testing for more details.

First, construct folders named models, outputs, test_data

### Data Preprocessing
This part is used to generate the sequential data for training.
```
python ./maml/preprocess.py --filename=/A/B.npz --cluster_file=/cluster/A/cluster_B --save_filename=B_seq.npz
```
A can be replaced by the task (taxi, bike, environment), B can be replaced by the city (e.g., nyc, dc)

### Training
For training, please use:
```
python ./maml/train_model.py --cities=several cities --save_dir=./models --model_type=att_metatrain_mem8 --update_batch_size=128 --test_num_updates=5 --threshold=0 --mem_dim=8 --cluster_loss_weight=1e-4 --meta_lr=1e-5 --update_lr=1e-5 --iterations=20000 --gpu_id=0
```

### Testing
For testing, please use:

```
python ./maml/test_model.py --city=chicago --save_dir=./models --output_dir=./outputs --model_type=att_metatrain_mem8 --test_model=model_3200 --test_days=3 --update_batch_size=128 --threshold=0 --meta_lr=1e-5 --update_lr=1e-5 --epochs=30 --gpu_id=0
```

Finally, run analysis.py to get denormalized results
