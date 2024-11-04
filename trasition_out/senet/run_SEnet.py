import subprocess
import time

epochs = [200,160,110,220,170,140,190,260,230,210,300,320,380,400,410,350]

commands = [
    f'CUDA_VISIBLE_DEVICES=3 EPOCH={epoch} python train.py' for epoch in epochs
]

times = []

for cmd in commands:
    start_time = time.time()
    subprocess.run(cmd, shell=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    times.append(elapsed_time)

for i, cmd in enumerate(commands):
    print(f"Command: {cmd}\nTime elapsed: {times[i]} seconds\n")