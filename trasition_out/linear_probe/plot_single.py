import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import font_manager as fm

def plot_predictions(predictions, actuals, title, filename):
    plt.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(15, 15))
    plt.scatter(actuals, predictions, label='Predicted vs Actual',s=135,edgecolors='black', linewidths=1.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='orange', label='Ideal Prediction')
    plt.xlabel('Actual Execution Time (s)', fontsize=40)
    plt.ylabel('Predicted Execution Time (s)', fontsize=40)
    plt.title(title,fontsize=25)
    plt.legend(fontsize=35)
    plt.grid(True)
    plt.savefig(filename,bbox_inches='tight')
    plt.show()

# test_predictions = np.array([16835, 17992, 26853, 17813])
# # , 4572, 10160, 27207, 27211
# test_actuals = np.array([21286, 20151, 31000, 12532])
# # , 6855, 11513, 12000, 15761
# test_pic_file = 'pic_result/our_model.png'

# 改过风格的真实值 4110, 4073, 6876, 6841, 28654, 28510, 48090, 60242
# change预测 43096, 27943, 75746, 50553, 58119, 75451, 98449, 46980
# modify预测  0, 4713, 13993, 14235, 27512, 32226, 26508, 46463

test_predictions = np.array([0, 4713, 13993, 14235, 27512, 32226, 26508, 46463])
test_predictions2 = np.array([43096, 27943, 75746, 50553, 58119, 75451, 98449, 46980])
# , 9667, 18263, 17400, 25111
test_actuals = np.array([4110, 4073, 6876, 6841, 28654, 28510, 48090, 60242])
# , 6855, 11513, 12000, 15761
test_pic_file = 'pic_result/test_modify.png'
test_pic_file2 = 'pic_result/test_change.png'

plot_predictions(test_predictions, test_actuals, 'Testing Data Predictions', test_pic_file)
plot_predictions(test_predictions2, test_actuals, 'Testing Data Predictions', test_pic_file2)

# ----------------------------------------------------------------------------
# baseline对比
# our:2353,10036,13248,11787,20562,13881,20664,10511,18615,23945,20562,13881,15749,18169,25145,23332,16835, 17992, 26853, 17813,4572, 10160, 27207,27211,12577

# true:2700,5915,7073,8213,12465,12040,12200,13510,28500,22984,15825,11520,28338,25618,26500,28969,21286, 20151, 31000, 12532,6855, 11513, 12000, 15761,9443

# baseline:3860,9125,9648,12140,21090,18250,17367,20234,48206,32804,25094,29656,40468,34734,45193,43343,36150,27067,47906,18212,9667,18263,17400,25111,13508

test_predictions2 = np.array([
    2353,10036,13248,11787,20562,13881,20664,10511,18615,23945,20562,13881,15749,18169,25145,23332,16835, 17992, 26853, 17813,4572, 10160, 27207,27211,12577
])
test_actuals = np.array([
    2700,5915,7073,8213,12465,12040,12200,13510,28500,22984,15825,11520,28338,25618,26500,28969,21286, 20151, 31000, 12532,6855, 11513, 12000, 15761,9443
])

test_predictions = np.array([
    3860,9125,9648,12140,21090,18250,17367,20234,48206,32804,25094,29656,40468,34734,45193,43343,36150,27067,47906,18212,9667,18263,17400,25111,13508
])

test_pic_file = 'pic_result/baseline_v100.png'
test_pic_file2 = 'pic_result/ourmodel_v100.png'

plot_predictions(test_predictions, test_actuals, 'Testing Data Predictions', test_pic_file)
plot_predictions(test_predictions2, test_actuals, 'Testing Data Predictions', test_pic_file2)




import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_comparative_predictions(predictions1, predictions2, actuals, title1, title2, filename):
    plt.rcParams.update(mpl.rcParamsDefault)  # Reset to default

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 12))  # Create 2x1 subplot

    # Plot for first set of predictions
    axes[0].scatter(actuals, predictions1, label='Predicted vs Actual')
    axes[0].plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='orange', label='Ideal Prediction')
    axes[0].set_xlabel('Actual Execution Time', fontsize=15)  # Set font size for x-axis label
    axes[0].set_ylabel('Predicted Execution Time', fontsize=15)  # Set font size for y-axis label
    axes[0].set_title(title1, fontsize=23)  # Set font size for title
    legend = axes[0].legend(fontsize=18)  # Set font size for legend
    axes[0].grid(True)

    # Plot for second set of predictions
    axes[1].scatter(actuals, predictions2, label='Predicted vs Actual')
    axes[1].plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='orange', label='Ideal Prediction')
    axes[1].set_xlabel('Actual Execution Time', fontsize=15)  # Set font size for x-axis label
    axes[1].set_ylabel('Predicted Execution Time', fontsize=15)  # Set font size for y-axis label
    axes[1].set_title(title2, fontsize=23)  # Set font size for title
    legend = axes[1].legend(fontsize=18)  # Set font size for legend
    axes[1].grid(True)

    plt.tight_layout()  # Adjust layout to not overlap
    plt.savefig(filename)
    plt.show()

# 改过风格的真实值 4110, 4073, 6876, 6841, 28654, 28510, 48090, 60242
# change预测 43096, 27943, 75746, 50553, 58119, 75451, 98449, 46980
# modify预测  0, 4713, 13993, 14235, 27512, 32226, 26508, 46463
predictions1 = np.array([43096, 27943, 75746, 50553, 58119, 75451, 98449, 46980])
predictions2 = np.array([0, 4713, 13993, 14235, 27512, 32226, 26508, 46463])
actuals = np.array([4110, 4073, 6876, 6841, 28654, 28510, 48090, 60242])

filename = 'pic_result/contradiction_rewrite.png'

# plot_comparative_predictions(predictions1, predictions2, actuals, "prediction before rewrite", "prediction after rewrite", filename)