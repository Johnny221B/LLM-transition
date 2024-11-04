import habitat
import torch
import torchvision.models as models
import torch.nn as nn

# Define VGG16 model
model = models.resnet50().cuda()
# print(model)
criterion = nn.CrossEntropyLoss()  # 损失函数

# 定义需要测试的 batch_size 和 epoch 组合的字典
test_combinations = {
    "test1": {"batch_size": 32, "epoch": 50},
    "test2": {"batch_size": 32, "epoch": 60},
    "test3": {"batch_size": 32, "epoch": 100},
    "test4": {"batch_size": 32, "epoch": 70},
    "test5": {"batch_size": 32, "epoch": 80},
    "test6": {"batch_size": 32, "epoch": 150},
    "test7": {"batch_size": 32, "epoch": 140},
    "test8": {"batch_size": 32, "epoch": 90},
    "test9": {"batch_size": 32, "epoch": 120},
    "test10": {"batch_size": 32, "epoch": 75}
    # 你可以根据需要添加更多的组合
}

# 定义结果字典来存储每个组合的运行时间
results_dict = {}

# 循环遍历每个组合
for test_name, params in test_combinations.items():
    batch_size = params["batch_size"]
    epoch = params["epoch"]

    # 生成随机输入
    image = torch.rand(batch_size, 3, 224, 224).cuda()  # 批量大小根据当前组合
    target = torch.randint(0, 1000, (batch_size,)).cuda()  # 随机生成标签

    tracker = habitat.OperationTracker(device=habitat.Device.V100)

    # 测量前向传播时间
    with tracker.track():
        out = model(image)
    trace = tracker.get_tracked_trace()
    forward_time_ms = trace.run_time_ms
    # print(f"Forward pass time for {test_name}: {forward_time_ms} ms")

    # 反向传播
    loss = criterion(out, target)
    with tracker.track():
        loss.backward()
    trace_back = tracker.get_tracked_trace()
    backward_time_ms = trace_back.run_time_ms
    # print(f"Backward pass time for {test_name}: {backward_time_ms} ms")

    # 预测时间
    pred1 = trace.to_device(habitat.Device.A100)
    pred2 = trace.to_device(habitat.Device.V100)
    pred1_back = trace_back.to_device(habitat.Device.A100)
    pred2_back = trace_back.to_device(habitat.Device.V100)

    # 计算总时间
    # total_a100 = (pred1.run_time_ms) * epoch * 50/batch_size
    # total_v100 = (pred2.run_time_ms) * epoch * 50/batch_size
    total_a100 = (pred1.run_time_ms + pred1_back.run_time_ms) * epoch * 50/batch_size
    total_v100 = (pred2.run_time_ms + pred2_back.run_time_ms) * epoch * 50/batch_size

    print(f"Total time on A100 for {test_name} (batch_size {batch_size}, epoch {epoch}): {total_a100} ms")
    print(f"Total time on V100 for {test_name} (batch_size {batch_size}, epoch {epoch}): {total_v100} ms")

    # 将结果存储到字典中
    results_dict[test_name] = {
        "batch_size": batch_size,
        "epoch": epoch,
        "A100_total_time_ms": total_a100,
        "V100_total_time_ms": total_v100,
    }

# 打印最终的结果字典
# print("Final results:")
# print(results_dict)
