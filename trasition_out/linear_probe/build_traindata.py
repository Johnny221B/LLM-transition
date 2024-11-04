import json
import re

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def clean_code(code):
    return re.sub(r'^\s*\n', '', code, flags=re.MULTILINE)

def format_code_files_to_json(file_paths, times, output_file):
    if len(file_paths) != len(times):
        raise ValueError("File paths and times lists must have the same length.")
    
    formatted_data = []
    
    for file_path, time in zip(file_paths, times):
        code = read_file(file_path)
        code = code.strip()
        code = clean_code(code)
        formatted_data.append({"code": code.strip(), "time": time})
    
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=4, sort_keys=True)

file_paths = [
    "/home/jingxuan/base_task/basic_task1/train_gan1.py",
    "/home/jingxuan/base_task/basic_task1/train_gan2.py",
    "/home/jingxuan/base_task/basic_task1/train_gan3.py",
    "/home/jingxuan/base_task/basic_task1/train_gan4.py",
    "/home/jingxuan/base_task/basic_task1/train_gan5.py",
    "/home/jingxuan/base_task/basic_task1/train_gan6.py",
    "/home/jingxuan/base_task/basic_task1/train_gan7.py",
    "/home/jingxuan/base_task/basic_task1/train_gan8.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet501.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet502.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet503.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet504.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet505.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet506.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet507.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet508.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT1.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT2.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT3.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT4.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT5.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT6.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT7.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT8.py",
    "/home/jingxuan/base_task/basic_task2/run_bert1.py",
    "/home/jingxuan/base_task/basic_task2/run_bert2.py",
    "/home/jingxuan/base_task/basic_task2/run_bert3.py",
    "/home/jingxuan/base_task/basic_task2/run_bert4.py",
    "/home/jingxuan/base_task/basic_task2/run_bert5.py",
    "/home/jingxuan/base_task/basic_task2/run_bert6.py",
    "/home/jingxuan/base_task/basic_task2/run_bert7.py",
    "/home/jingxuan/base_task/basic_task2/run_bert8.py"
]

A100_time = [
    4266,1265,3038,3473,5972,4611,5076,4457,
    6411,9913,7156,8210,15065,14154,9215,12875,
    24755,41468,33326,22834,45839,49361,18518,37383,
    19992,23238,30075,53525,66800,49582,72529,43160
]

A100_energy = [
    199994,61200,145477,166085,283440,220361,242772,215950,
    1141170,1747399,1262944,1453539,2674768,2529022,1652184,2297022,
    5724971,9624660,7711786,5290215,10603099,11390482,4284986,8567837,
    4597310,5359083,6943332,12316557,15411728,11326983,16624045,10051758
]

A6000_time = [
    7700,2380,5950,7600,12300,9650,10800,9600,
    7541,12565,8836,10060,19160,17900,11460,15100,
    23000,37800,31000,21100,42200,45600,16900,33800,
    22000,26000,33000,68700,85000,63700,80600,48000
]

A6000_energy = [
    191140,59125,148062,189076,305630,239266,266878,238401,
    186986,311794,219852,250282,476106,443454,283205,374993,
    571676,940019,766715,524793,1056930,1120848,414814,828936,
    546711,643773,819406,1721707,2089767,1562815,1977017,1177764
]

# format_code_files_to_json(file_paths, A100_time, 'train_A100_time.json')
# format_code_files_to_json(file_paths, A6000_time, 'train_A6000_time.json')
# format_code_files_to_json(file_paths, A100_energy, 'train_A100_energy.json')
# format_code_files_to_json(file_paths, A6000_energy, 'train_A6000_energy.json')

file_paths_small = [
    "/home/jingxuan/base_task/basic_task1/train_resnet501.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet502.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet503.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet504.py",
    "/home/jingxuan/base_task/basic_task2/run_bert1.py",
    "/home/jingxuan/base_task/basic_task2/run_bert2.py",
    "/home/jingxuan/base_task/basic_task2/run_bert3.py",
    "/home/jingxuan/base_task/basic_task2/run_bert4.py",
    "/home/jingxuan/base_task/basic_task1/train_gan1.py",
    "/home/jingxuan/base_task/basic_task1/train_gan2.py",
    "/home/jingxuan/base_task/basic_task1/train_gan3.py",
    "/home/jingxuan/base_task/basic_task1/train_gan4.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT1.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT2.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT3.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT4.py"
]
times_small = [
        7541.934, 12565.417, 8836.583, 10061.487,
        22000,26000,33000,68700,
        7700,2380,5950,7600,
        23053.283, 37774.016, 31033.167, 21068.562
]

format_code_files_to_json(file_paths_small, times_small, 'trainSmall_dataset.json')

file_paths2 = [
    "/home/jingxuan/base_task/basic_task1/train_resnet501.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet502.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet503.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet504.py",
    "/home/jingxuan/base_task/basic_task2/run_bert1.py",
    "/home/jingxuan/base_task/basic_task2/run_bert2.py",
    "/home/jingxuan/base_task/basic_task2/run_bert3.py",
    "/home/jingxuan/base_task/basic_task2/run_bert4.py",
    "/home/jingxuan/base_task/basic_task2/run_bert5.py",
    "/home/jingxuan/base_task/basic_task2/run_bert6.py",
    "/home/jingxuan/base_task/basic_task2/run_bert7.py",
    "/home/jingxuan/base_task/basic_task2/run_bert8.py",
    "/home/jingxuan/base_task/basic_task1/train_gan1.py",
    "/home/jingxuan/base_task/basic_task1/train_gan2.py",
    "/home/jingxuan/base_task/basic_task1/train_gan3.py",
    "/home/jingxuan/base_task/basic_task1/train_gan4.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT1.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT2.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT3.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT4.py"
]

times2 = [
        7541.934, 12565.417, 8836.583, 10061.487,
        22000,26000,33000,68700,85000,63700,80600,48000,
        7700,2380,5950,7600,
        23053.283, 37774.016, 31033.167, 21068.562
]
format_code_files_to_json(file_paths2, times2, 'trainReinforce_dataset.json')