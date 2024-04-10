# -*- coding: utf-8 -*-

import os
import argparse

import traceback
import os.path
from text.cleaner import clean_text
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


parser = argparse.ArgumentParser(description="GPT-SoVITS tool")

parser.add_argument("-it", "--inp_text", type=str, default="", help="asr 文件地址")
parser.add_argument("-iad", "--inp_audio_dir", type=str, default="", help="音频文件目录")
parser.add_argument("-m", "--model_name", type=str, default="", help="模型名称")
parser.add_argument("-i", "--i_part", type=str, default="", help="part")
parser.add_argument("-pts", "--all_parts", type=str, default="", help="all_parts")
parser.add_argument("-o", "--opt_dir", type=str, default="", help="opt_dir")
parser.add_argument("-btd", "--bert_pretrained_dir", type=str, default="", help="bert_pretrained_dir")
parser.add_argument("-ih", "--is_half", type=str, default="", help="is_half")
parser.add_argument("-g", "--gpus", type=str, default="0", help="gpus")
args = parser.parse_args()

inp_text = args.inp_text
inp_wav_dir = args.inp_audio_dir
exp_name = args.model_name
i_part = args.i_part
all_parts = args.all_parts
opt_dir = args.opt_dir
bert_pretrained_dir = args.bert_pretrained_dir
is_half = eval(args.is_half)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


from time import time as ttime
import shutil


def my_save(fea,path):#####fix issue: torch.save doesn't support chinese path
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path="%s%s.pth"%(ttime(),i_part)
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))


txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
if os.path.exists(txt_path) == False:
    bert_dir = "%s/3-bert" % (opt_dir)
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(bert_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)
    if is_half == True:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)

    def get_bert_feature(text, word2ph):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)
            res = bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)

        return phone_level_feature.T

    def process(data, res):
        for name, text, lan in data:
            try:
                name = os.path.basename(name)
                phones, word2ph, norm_text = clean_text(
                    text.replace("%", "-").replace("￥", ","), lan
                )
                path_bert = "%s/%s.pt" % (bert_dir, name)
                if os.path.exists(path_bert) == False and lan == "zh":
                    bert_feature = get_bert_feature(norm_text, word2ph)
                    assert bert_feature.shape[-1] == len(phones)
                    # torch.save(bert_feature, path_bert)
                    my_save(bert_feature, path_bert)
                phones = " ".join(phones)
                # res.append([name,phones])
                res.append([name, phones, word2ph, norm_text])
            except:
                print(name, text, traceback.format_exc())

    todo = []
    res = []
    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    language_v1_to_language_v2 = {
        "ZH": "zh",
        "zh": "zh",
        "JP": "ja",
        "jp": "ja",
        "JA": "ja",
        "ja": "ja",
        "EN": "en",
        "en": "en",
        "En": "en",
    }
    for line in lines[int(i_part) :: int(all_parts)]:
        try:
            wav_name, spk_name, language, text = line.split("|")
            # todo.append([name,text,"zh"])
            todo.append(
                [wav_name, text, language_v1_to_language_v2.get(language, language)]
            )
        except:
            print(line, traceback.format_exc())

    process(todo, res)
    opt = []
    for name, phones, word2ph, norm_text in res:
        opt.append("%s\t%s\t%s\t%s" % (name, phones, word2ph, norm_text))
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
