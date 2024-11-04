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
    "/home/jingxuan/base_task/basic_task1/train_gan.py",
    "/home/jingxuan/base_task/basic_task1/train_gan9.py",
    "/home/jingxuan/base_task/basic_task1/train_gan_style1.py",
    "/home/jingxuan/base_task/basic_task1/train_gan_style2.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet50.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet509.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet_style1.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet_style2.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT9.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT_style1.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT_style2.py",
    "/home/jingxuan/base_task/basic_task2/run_bert.py",
    "/home/jingxuan/base_task/basic_task2/run_bert9.py",
    "/home/jingxuan/base_task/basic_task2/run_bert_style1.py",
    "/home/jingxuan/base_task/basic_task2/run_bert_style2.py"
    # "/home/jingxuan/base_task/basic_task1/train_ViT_style1.py",
    # "/home/jingxuan/base_task/basic_task1/train_ViT_style2.py",
    # "/home/jingxuan/base_task/basic_task2/run_bert_style1.py",
    # "/home/jingxuan/base_task/basic_task2/run_bert_style2.py",
    # "/home/jingxuan/base_task/basic_task1/train_resnet_style1.py",
    # "/home/jingxuan/base_task/basic_task1/train_resnet_style2.py",
    # "/home/jingxuan/base_task/basic_task1/train_gan_style3.py",
    # "/home/jingxuan/base_task/basic_task1/train_gan_style4.py",
]

times_A6000 = [
    4550,7150,8080,7950,
    6680,9450,8870,8838,
    19300,28200,27154,26887,
    44500,52000,61200,65610
]

times_A100 = [
    2209,3332,4110,4073,
    5290,7718,6876,6841,
    20692,31495,28654,28510,
    40600,46486,48090,60242
]

energy_A6000 = [
    112805,178052,198622,195116,
    165724,235380,220401,219770,
    478620,692295,670228,664962,
    1106614,1272518,1512695,1623617
]

energy_A100 = [
    104511,160069,199924,199053,
    946317,1377956,1308705,1253195,
    4803658,7197166,6647283,6556266,
    9298090,10728771,11480465,14237323
]

format_code_files_to_json(file_paths, times_A6000, 'test_A6000_time.json')
format_code_files_to_json(file_paths, energy_A6000, 'test_A6000_energy.json')
format_code_files_to_json(file_paths, times_A100, 'test_A100_time.json')
format_code_files_to_json(file_paths, energy_A100, 'test_A100_energy.json')
# 示例文件路径列表
file_paths2 = [
    "/home/jingxuan/base_task/basic_task1/train_gan.py",
    "/home/jingxuan/base_task/basic_task1/train_gan9.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet50.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet509.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT9.py",
    "/home/jingxuan/base_task/basic_task2/run_bert.py",
    "/home/jingxuan/base_task/basic_task2/run_bert9.py"
]

# 示例时间值列表
timesindist_A6000 = [
        4550, 7150,
        6680, 9450,
        19304, 28200,
        44500, 52000
]

timesindist_A100 = [
    2209,3332,
    5290,7718,
    20692,31495,
    40600,46486
]

energyindist_A6000 = [
    112805,178052,
    165724,235380,
    478620,692295,
    1106614,1272518
]

energyindist_A100 = [
    104511,160069,
    946317,1377956,
    4803658,7197166,
    9298090,10728771
]

format_code_files_to_json(file_paths2, timesindist_A6000, 'testindist_A6000_time.json')
format_code_files_to_json(file_paths2, energyindist_A6000, 'testindist_A6000_energy.json')
format_code_files_to_json(file_paths2, timesindist_A100, 'testindist_A100_time.json')
format_code_files_to_json(file_paths2, energyindist_A100, 'testindist_A100_energy.json')

# 使用示例
# format_code_files_to_json(file_paths2, times2, 'test_dataset2.json')

file_paths3 = [
    "/home/jingxuan/base_task/basic_task1/train_gan_style1.py",
    "/home/jingxuan/base_task/basic_task1/train_gan_style2.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet_style1.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet_style2.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT_style1.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT_style2.py",
    "/home/jingxuan/base_task/basic_task2/run_bert_style1.py",
    "/home/jingxuan/base_task/basic_task2/run_bert_style2.py"
]

times3_A6000 = [
    8080,7950,
    8870,8838,
    27154,26887,
    61200,65610
]

times3_A100 = [
4110,4073,
6876,6841,
28654,28510,
48090,60242
]

energy3_A6000 = [
    198622,195116,
    220401,219770,
    670228,664962,
    1512695,1623617
]

energy3_A100 = [
    199924,199053,
    1308705,1253195,
    6647283,6556266,
    11480465,14237323
]

file_paths4 = [
    "/home/jingxuan/base_task/basic_task1/train_gan_style1_modify.py",
    "/home/jingxuan/base_task/basic_task1/train_gan_style2_modify.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet_style1_modify.py",
    "/home/jingxuan/base_task/basic_task1/train_resnet_style2_modify.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT_style1_modify.py",
    "/home/jingxuan/base_task/basic_task1/train_ViT_style2_modify.py",
    "/home/jingxuan/base_task/basic_task2/run_bert_style1_modify.py",
    "/home/jingxuan/base_task/basic_task2/run_bert_style2_modify.py"
]

format_code_files_to_json(file_paths3, times3_A6000, 'testchange_A6000_time.json')
format_code_files_to_json(file_paths3, energy3_A6000, 'testchange_A6000_energy.json')
format_code_files_to_json(file_paths3, times3_A100, 'testchange_A100_time.json')
format_code_files_to_json(file_paths3, energy3_A100, 'testchange_A100_energy.json')

format_code_files_to_json(file_paths4, times3_A6000, 'testmodify_A6000_time.json')
format_code_files_to_json(file_paths4, energy3_A6000, 'testmodify_A6000_energy.json')
format_code_files_to_json(file_paths4, times3_A100, 'testmodify_A100_time.json')
format_code_files_to_json(file_paths4, energy3_A100, 'testmodify_A100_energy.json')

file_paths_VGG = [
    "/home/jingxuan/base_task/LLM-VGG/train_VGG16.py",
    "/home/jingxuan/base_task/LLM-VGG/train_VGG17.py",
    "/home/jingxuan/base_task/LLM-VGG/train_VGG18.py",
    "/home/jingxuan/base_task/LLM-VGG/train_VGG19.py",
    "/home/jingxuan/base_task/LLM-VGG/train_VGG1_style1.py",
    "/home/jingxuan/base_task/LLM-VGG/train_VGG2_style1.py",
    "/home/jingxuan/base_task/LLM-VGG/train_VGG1_style2.py",
    "/home/jingxuan/base_task/LLM-VGG/train_VGG2_style2.py"
]

times_VGG = [
    21286,20151,31000,12532,
    6855,11513,12000,15761
]

format_code_files_to_json(file_paths_VGG, times_VGG, 'test_datasetVGG.json')