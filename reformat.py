import json

# 读取原始 JSON 文件
with open('/data1_8t/user/cmx/Sa2VA-main/NTIRE/results/result_15mlp3.json', 'r') as file:
    data = json.load(file)

# 递归函数来修改所有的 "element_result" 键
def rename_key(d):
    if isinstance(d, dict):
        # 修改键值
        if "element_result" in d:
            d["element_score"] = d.pop("element_result")
        # 递归修改字典中的其他值
        for key, value in d.items():
            rename_key(value)
    elif isinstance(d, list):
        # 对列表中的每个元素递归调用
        for item in d:
            rename_key(item)

# 修改数据
rename_key(data)

# 保存修改后的 JSON 数据
with open('/data1_8t/user/cmx/Sa2VA-main/NTIRE/results/output_15mlp3.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Key 'element_result' has been changed to 'element_score'.")