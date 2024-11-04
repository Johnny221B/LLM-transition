import json
import os

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def format_code_files_to_json(file_paths, times, output_file):
    if len(file_paths) != len(times):
        raise ValueError("File paths and times lists must have the same length.")
    
    formatted_data = []
    for file_path, time in zip(file_paths, times):
        code = read_file(file_path)
        code = code.strip()
        code = code.replace('n','\\n')
        formatted_data.append({"code": code.strip(), "time": time})
    
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=4)

tasks = ['resnet','resnet2','resnet3','bert','bert2','bert3','gan','gan2','gan3','vit2','vit3','vit','vgg','vgg2','vgg3']
resnet_path = ["/home/jingxuan/base_task/basic_task1/train_resnet50.py"]
resnet_path2 = ["/home/jingxuan/base_task/basic_task1/train_resnet_style1.py"]
resnet_path3 = ["/home/jingxuan/base_task/basic_task1/train_resnet_style2.py"]
bert_path = ["/home/jingxuan/base_task/basic_task2/run_bert.py"]
bert_path2 = ["/home/jingxuan/base_task/basic_task2/run_bert_style1.py"]
bert_path3 = ["/home/jingxuan/base_task/basic_task2/run_bert_style2.py"]
gan_path = ["/home/jingxuan/base_task/basic_task1/train_gan.py"]
gan_path2 = ["/home/jingxuan/base_task/basic_task1/train_gan_style1.py"]
gan_path3 = ["/home/jingxuan/base_task/basic_task1/train_gan_style2.py"]
vit_path = ["/home/jingxuan/base_task/basic_task1/train_ViT.py"]
vit_path2 = ["/home/jingxuan/base_task/basic_task1/train_ViT_style1.py"]
vit_path3 = ["/home/jingxuan/base_task/basic_task1/train_ViT_style2.py"]
vgg_path = ["/home/jingxuan/base_task/LLM-VGG/train_VGG0.py"]
vgg_path2 = ["/home/jingxuan/base_task/LLM-VGG/train_VGG1_style1.py"]
vgg_path3 = ["/home/jingxuan/base_task/LLM-VGG/train_VGG1_style2.py"]


task_paths = {
    'resnet': resnet_path,
    'resnet2': resnet_path2,
    'resnet3': resnet_path3,
    'bert': bert_path,
    'bert2': bert_path2,
    'bert3': bert_path3,
    'gan': gan_path,
    'gan2': gan_path2,
    'gan3': gan_path3,
    'vit': vit_path,
    'vit2': vit_path2,
    'vit3': vit_path3,
    'vgg': vgg_path,
    'vgg2': vgg_path2,
    'vgg3': vgg_path3
}
times = [0]

output_dir = '/home/jingxuan/linear_probing/validation'  # 指定存放JSON文件的路径

# 执行每个任务并生成对应的JSON文件
for task in tasks:
    file_paths = task_paths[task]
    output_file = os.path.join(output_dir, f'{task}_val.json')  # 生成对应文件名
    format_code_files_to_json(file_paths, times, output_file)  # 执行JSON文件生成
    print(f'{output_file} 已生成并保存到 {output_dir}')