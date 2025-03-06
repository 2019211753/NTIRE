import json
import csv
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='right')
tokenizer.add_special_tokens({"bos_token": "[DEC]"})
def is_sublist(lst1, lst2):
    return str(lst1)[1:-1] in str(lst2)[1:-1]

def load_csv_as_dict_list(file_path):
    dict_list = []
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            dict_list.append(row)
    return dict_list

def load_json_as_dict_list(file_path):
    with open(file_path, mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    return data

def load_data(file_path, file_type):
    if file_type == 'csv':
        return load_csv_as_dict_list(file_path)
    elif file_type == 'json':
        return load_json_as_dict_list(file_path)
    else:
        raise ValueError("Unsupported file type. Please use 'csv' or 'json'.")
    
data = load_data('dataset/train.json','json')

def get_index(list1,list2):
    len_list1 = len(list1)
    len_list2 = len(list2)
    for i in range(len_list2 - len_list1 + 1):
        if list2[i:i + len_list1] == list1:
            return i
    return 0

def is_sublist(lst1, lst2):
    return str(lst1)[1:-1] in str(lst2)[1:-1]
# max_len = 0
# for item in tqdm(data):
#     prompt = item['prompt']
#     ids = tokenizer(prompt).input_ids
#     if max_len < len(ids):
#         max_len = len(ids)
#     breakpoint()
# print(max_len)

error = 0
data_new = []
data_error = {}
len_flow = 0
for item in tqdm(data):
    elements = item['element_score'].keys()
    prompt = item['prompt']
    prompt_ids = tokenizer(prompt).input_ids
    mask = np.array([0] * len(prompt_ids))
    element_score = np.array([0] * len(prompt_ids),dtype=np.float32)
    if len(prompt_ids) > 32:
        len_flow += 1
    flag = 1
    for element in elements:
        element_ = element.rpartition('(')[0]
        element_ids = tokenizer(element_).input_ids[1:-1]
        # breakpoint()

        idx = get_index(element_ids,prompt_ids)
        # breakpoint()
        if idx == 0:
            # breakpoint()
            # print(prompt)
            if prompt not in data_error:
                data_error[prompt] = {}
            data_error[prompt][element] = None
            error += 1
            flag = 0 
            # break
        else:
            mask[idx:idx+len(element_ids)] = mask[idx:idx+len(element_ids)] + 1
            element_score[idx:idx+len(element_ids)] = element_score[idx:idx+len(element_ids)] + item['element_score'][element]
            # print(element_score)
            # breakpoint()

    index = np.where(mask != 0)
    element_score[index] = element_score[index] / mask[index].astype(np.float32)
    mask[index] = 1       

    
    if flag:
        # breakpoint()
        item['mask'] = mask.tolist()
        item['token_score'] = element_score.tolist()
        data_new.append(item)
    # else:
    #     data_error.append(item)

print(len_flow)
print(len(data))
print(error)
with open('dataset/train_mask.json', 'w', newline='', encoding='utf-8') as file:
    json.dump(data_new, file, ensure_ascii=False, indent=4)

# with open('dataset/data_train_error.json', 'w', newline='', encoding='utf-8') as file:
#     json.dump(data_error, file, ensure_ascii=False, indent=4)


    


