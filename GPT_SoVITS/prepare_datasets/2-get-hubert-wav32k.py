# -*- coding: utf-8 -*-

import sys,os
import argparse
from feature_extractor import cnhubert
import traceback,numpy as np,logging
from scipy.io import wavfile
import librosa,torch
now_dir = os.getcwd()
sys.path.append(now_dir)
from my_utils import load_audio

parser = argparse.ArgumentParser(description="GPT-SoVITS tool")
parser.add_argument("-it", "--inp_text", type=str, default="", help="asr 文件地址")
parser.add_argument("-iad", "--inp_audio_dir", type=str, default="", help="音频文件目录")
parser.add_argument("-m", "--model_name", type=str, default="", help="模型名称")
parser.add_argument("-i", "--i_part", type=str, default="", help="part")
parser.add_argument("-pts", "--all_parts", type=str, default="", help="all_parts")
parser.add_argument("-o", "--opt_dir", type=str, default="", help="opt_dir")
parser.add_argument("-cnd", "--cnhubert_base_dir", type=str, default="", help="cnhubert_base_dir")
parser.add_argument("-ih", "--is_half", type=str, default="", help="is_half")
parser.add_argument("-g", "--gpus", type=str, default="0", help="gpus")
args = parser.parse_args()

inp_text = args.inp_text
inp_wav_dir = args.inp_audio_dir
exp_name = args.model_name
i_part = args.i_part
all_parts = args.all_parts
opt_dir = args.opt_dir
cnhubert.cnhubert_base_path = args.cnhubert_base_dir
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

hubert_dir="%s/4-cnhubert"%(opt_dir)
wav32dir="%s/5-wav32k"%(opt_dir)
os.makedirs(opt_dir,exist_ok=True)
os.makedirs(hubert_dir,exist_ok=True)
os.makedirs(wav32dir,exist_ok=True)

maxx=0.95
alpha=0.5
if torch.cuda.is_available():
    device = "cuda:0"
# elif torch.backends.mps.is_available():
#     device = "mps"
else:
    device = "cpu"
model=cnhubert.get_model()
# is_half=False
if(is_half==True):
    model=model.half().to(device)
else:
    model = model.to(device)

nan_fails=[]
def name2go(wav_name,wav_path):
    hubert_path="%s/%s.pt"%(hubert_dir,wav_name)
    if(os.path.exists(hubert_path)):return
    tmp_audio = load_audio(wav_path, 32000)
    tmp_max = np.abs(tmp_audio).max()
    if tmp_max > 2.2:
        print("%s-filtered,%s" % (wav_name, tmp_max))
        return
    tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha*32768)) + ((1 - alpha)*32768) * tmp_audio
    tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha*1145.14)) + ((1 - alpha)*1145.14) * tmp_audio
    tmp_audio = librosa.resample(
        tmp_audio32b, orig_sr=32000, target_sr=16000
    )#不是重采样问题
    tensor_wav16 = torch.from_numpy(tmp_audio)
    if (is_half == True):
        tensor_wav16=tensor_wav16.half().to(device)
    else:
        tensor_wav16 = tensor_wav16.to(device)
    ssl=model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1,2).cpu()#torch.Size([1, 768, 215])
    if np.isnan(ssl.detach().numpy()).sum()!= 0:
        nan_fails.append(wav_name)
        print("nan filtered:%s"%wav_name)
        return
    wavfile.write(
        "%s/%s"%(wav32dir,wav_name),
        32000,
        tmp_audio32.astype("int16"),
    )
    my_save(ssl,hubert_path )

with open(inp_text,"r",encoding="utf8")as f:
    lines=f.read().strip("\n").split("\n")

for line in lines[int(i_part)::int(all_parts)]:
    try:
        # wav_name,text=line.split("\t")
        wav_name, spk_name, language, text = line.split("|")
        if (inp_wav_dir != "" and inp_wav_dir != None):
            wav_name = os.path.basename(wav_name)
            wav_path = "%s/%s"%(inp_wav_dir, wav_name)

        else:
            wav_path=wav_name
            wav_name = os.path.basename(wav_name)
        name2go(wav_name,wav_path)
    except:
        print(line,traceback.format_exc())

if(len(nan_fails)>0 and is_half==True):
    is_half=False
    model=model.float()
    for wav_name in nan_fails:
        try:
            name2go(wav_name)
        except:
            print(wav_name,traceback.format_exc())
