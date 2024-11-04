import re
import glob

def remove_comments(source_text):
    # 删除多行注释
    text_no_multiline = re.sub(r"(''')[\s\S]*?\1", "", source_text, flags=re.MULTILINE)
    text_no_multiline = re.sub(r'("""[\s\S]*?""")', '', text_no_multiline, flags=re.MULTILINE)
    # 删除单行注释
    text_no_comments = re.sub(r'#.*$', "", text_no_multiline, flags=re.MULTILINE)
    return text_no_comments

# 指定文件夹路径和文件命名模式
folder_path = '/home/jingxuan/base_task/basic_task1'
# file_pattern = "run_bert*.py"
file_pattern = "train_*.py"

# 使用 glob.glob 来获取所有匹配文件
files = glob.glob(f"{folder_path}/{file_pattern}")

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    new_content = remove_comments(content)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(new_content)

    print(f"Comments removed from {file_path}.")
