import numpy as np
import os
from sklearn.metrics import mean_squared_error

output_dir = './outputs/'
# output_dir = './outputs/bike_baselines_chicago/7/'

# output_dir = './outputs/bike_nyc+dc--new_boston/7/'
# output_dir = './outputs/nyc+dc+porto--boston/7/'

# output_dir = './outputs/bike_nyc+dc--chicago/3/'
# output_dir = './outputs/nyc+dc+porto--chicago/3/'

resdir = os.listdir(output_dir)
print(resdir)
threshold = 0
data = {}

# max_val = 334   # chicago
max_val = 164   # chicago

# max_val = 42   # new_boston
# max_val = 359   # boston

for eachfile in resdir:
    if 'output' in eachfile:
        data[eachfile] = np.load(output_dir + '{0}'.format(eachfile))['arr_0'].squeeze()
        data[eachfile] *= max_val
        print(eachfile, data[eachfile].shape)

print("")
labels = data["output_oracle.npz"]
norm_threshold = threshold

if 'boston' in output_dir and 'bike' not in output_dir:
    all_rmse = []
    for eachfile in data:
        if eachfile == 'output_oracle.npz':
            continue
        valid = labels >= norm_threshold
        rmse = np.sqrt(mean_squared_error(data['output_oracle.npz'][valid], data[eachfile][valid]))
        print(eachfile, rmse)
        all_rmse.append(rmse)

    print("")
    for i in range(len(all_rmse)):
        for j in range(i + 1, len(all_rmse)):
            print((all_rmse[j] - all_rmse[i]) / all_rmse[j])
else:
    for k in range(2):
        all_rmse = []
        for eachfile in data:
            if eachfile == 'output_oracle.npz':
                continue
            valid = labels[:, k] >= norm_threshold
            rmse = np.sqrt(mean_squared_error(data['output_oracle.npz'][valid, k], data[eachfile][valid, k]))
            print(eachfile, rmse)
            all_rmse.append(rmse)

        print("")
        for i in range(len(all_rmse)):
            for j in range(i+1, len(all_rmse)):
                print((all_rmse[j] - all_rmse[i]) / all_rmse[j])
