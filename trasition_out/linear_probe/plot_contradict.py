import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def plot_predictions(test_actuals1, test_predictions1, test_actuals2, test_predictions2, title, filename):
    plt.rcParams.update(matplotlib.rcParamsDefault)  # Reset to default
    plt.rcParams.update({'font.size': 28})
    plt.figure(figsize=(15, 15))
    
    # 绘制测试数据集的预测值和实际值
    plt.scatter(test_actuals1, test_predictions1, label='after reinforce', color='red', s=120)
    plt.scatter(test_actuals2, test_predictions2, label='before reinforce', color='purple',s=120)
    
    # 绘制完美预测的对角线
    all_actuals = np.concatenate([test_actuals1, test_actuals2])
    plt.plot([min(all_actuals), max(all_actuals)], 
             [min(all_actuals), max(all_actuals)], 
             color='orange', label='Perfect Prediction')
    
    # 在具有相同整数部分的实际时间的点之间绘制虚线
    unique_actuals = np.unique(np.floor(all_actuals))
    for actual in unique_actuals:
        test_pred1 = test_predictions1[np.floor(test_actuals1) == actual]
        test_pred2 = test_predictions2[np.floor(test_actuals2) == actual]
        
        if len(test_pred1) > 0 and len(test_pred2) > 0:
            plt.plot([actual, actual], [test_pred1[0], test_pred2[0]], 'k--', linewidth=0.5)
    
    plt.xlabel('Actual Running Time')
    plt.ylabel('Predicted Running Time')
    plt.title(title)
    plt.legend(fontsize=35)
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


train_predictions = np.array([12132.014,11125.418,48348.156,40057.074,1596.7067,2108.0874,30607.803,31037.855])

train_actuals = np.array([6680.7017,9445.0,39173.0,45767.0,916.04456,1571.9996,19304.295,28168.0])

train_predictions2 = np.array([11199.296,7080.885,26065.127,30910.992,2331.079,1696.8076,29826.312,20602.87])

train_actuals2 = np.array([6680.7017,9445.0,39173.0,45767.0,916.04456,1571.9996,19304.295,28168.0])

plot_predictions(train_actuals,train_predictions,train_actuals2,train_predictions2,"Contradiction between two sets","reinforce.png")

# 改过风格的真实值 
# 真实 4073, 4110, 6874, 6841, 28654, 28510, 48090, 60242
# change预测 43096, 27943, 75746, 50553, 58119, 75451, 98449, 46980
# modify预测  0, 4713, 13993, 14235, 27512, 32226, 26508, 46463

# predict: 12132.014,11125.418,48348.156,40057.074,1596.7067,2108.0874,30607.803,34037.855
# actuals: 6680.7017,9445.0,39173.0,45767.0,916.04456,1571.9996,19304.295,28168.0


# predict: 11199.296,7080.885,26065.127,30910.992,2331.079,1696.8076,29826.312,20602.87
# actuals: 6680.702,9445.0,39234.625,45767.0,916.04205,1571.999,19304.295,28168.0