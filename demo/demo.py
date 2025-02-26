from lavis.models import load_model_and_preprocess
import torch
from PIL import Image
from transformers import BertTokenizer
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='right')
tokenizer.add_special_tokens({"bos_token": "[DEC]"})

def get_index(list1,list2):
    len_list1 = len(list1)
    len_list2 = len(list2)
    for i in range(len_list2 - len_list1 + 1):
        if list2[i:i + len_list1] == list1:
            return i
    return 0

def element_score(prompt, img_path, element):
    img = Image.open(img_path).convert('RGB')
    img = vis_processors["eval"](img).unsqueeze(0).to(device)
    txt = text_processors["eval"](prompt)
    torch.cuda.empty_cache()
    with torch.no_grad():
        alignment_score, scores = model.element_score(img, [txt])
    
    prompt_ids = tokenizer(txt).input_ids
    element_ids = tokenizer(element).input_ids[1:-1]
    idx = get_index(element_ids, prompt_ids)
    if idx:
        mask = [0] * len(prompt_ids)
        mask[idx:idx+len(element_ids)] = [1] * len(element_ids)
        
        mask = torch.tensor(mask).to(device)

        element_score = (scores * mask).sum() / mask.sum().item()
        return alignment_score, element_score
    else:
        return -1
        
if __name__ == '__main__':
    model_path = 'checkpoints/fga_blip2.pth'

    prompt = 'A photograph of a lady practicing yoga in a quiet studio, full shot.'
    img_path = 'demo/demo.png'

    model, vis_processors, text_processors = load_model_and_preprocess("fga_blip2", "coco", device=device, is_eval=True)
    model.load_checkpoint(model_path)
    img = Image.open(img_path).convert('RGB')
    img = vis_processors["eval"](img).unsqueeze(0).to(device)
    txt = text_processors["eval"](prompt)

    itm_scores= model({"image": img, "text_input": txt}, match_head="itm",inference=True)
    # itm_scores = torch.nn.functional.softmax(itm_output, dim=1) * 4 + 1 
    overall_score = itm_scores.item()
    print('overall score: ', overall_score)

    alignment_score, element_scores = model.element_score(img, txt)
    print('elements score: ', element_scores)

    alignment_score, element_score = element_score(prompt, img_path, 'a lady')
    print('element "a lady" score: ', element_score)
