import os
import argparse

import traceback
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging, librosa, utils, torch
from module.models import SynthesizerTrn

logging.getLogger("numba").setLevel(logging.WARNING)

parser = argparse.ArgumentParser(description="GPT-SoVITS tool")
parser.add_argument("-it", "--inp_text", type=str, default="", help="asr 文件地址")
parser.add_argument("-m", "--model_name", type=str, default="", help="模型名称")
parser.add_argument("-i", "--i_part", type=str, default="", help="part")
parser.add_argument("-pts", "--all_parts", type=str, default="", help="all_parts")
parser.add_argument("-o", "--opt_dir", type=str, default="", help="opt_dir")
parser.add_argument("-ps", "--pretrained_s2G", type=str, default="", help="pretrained_s2G")
parser.add_argument("-sp", "--s2config_path", type=str, default="", help="s2config_path")
parser.add_argument("-ih", "--is_half", type=str, default="", help="is_half")
parser.add_argument("-g", "--gpus", type=str, default="0", help="gpus")
args = parser.parse_args()

inp_text = args.inp_text
exp_name = args.model_name
i_part = args.i_part
all_parts = args.all_parts
opt_dir = args.opt_dir
pretrained_s2G = args.pretrained_s2G
s2config_path = args.s2config_path
is_half = eval(args.is_half)
gpus = args.gpus


hubert_dir = "%s/4-cnhubert" % (opt_dir)
semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
if os.path.exists(semantic_path) == False:
    os.makedirs(opt_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda:" + gpus
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    hps = utils.get_hparams_from_file(s2config_path)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    # utils.load_checkpoint(utils.latest_checkpoint_path(hps.s2_ckpt_dir, "G_*.pth"), vq_model, None, True)
    # utils.load_checkpoint(pretrained_s2G, vq_model, None, True)
    print(
        vq_model.load_state_dict(
            torch.load(pretrained_s2G, map_location="cpu")["weight"], strict=False
        )
    )

    def name2go(wav_name, lines):
        hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
        if os.path.exists(hubert_path) == False:
            return
        ssl_content = torch.load(hubert_path, map_location="cpu")
        if is_half == True:
            ssl_content = ssl_content.half().to(device)
        else:
            ssl_content = ssl_content.to(device)
        codes = vq_model.extract_latent(ssl_content)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        lines.append("%s\t%s" % (wav_name, semantic))

    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    lines1 = []
    for line in lines[int(i_part) :: int(all_parts)]:
        # print(line)
        try:
            # wav_name,text=line.split("\t")
            wav_name, spk_name, language, text = line.split("|")
            wav_name = os.path.basename(wav_name)
            # name2go(name,lines1)
            name2go(wav_name, lines1)
        except:
            print(line, traceback.format_exc())
    with open(semantic_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines1))
