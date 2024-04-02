import os
import argparse
import traceback
import logging
from tools.i18n.i18n import I18nAuto
i18n = I18nAuto()

logger = logging.getLogger(__name__)
import librosa,ffmpeg
import torch
from mdxnet import MDXNetDereverb
from vr import AudioPre, AudioPreDeEcho

weight_uvr5_root = "tools/uvr5/uvr5_weights"

parser = argparse.ArgumentParser(description="GPT-SoVITS uvr5 tool")
parser.add_argument("", "--model_name", type=str, default="HP2_all_vocals", help="model_name")
parser.add_argument("", "--inp_root", type=str, default="", help="inp_root")
parser.add_argument("", "--opt_root", type=str, default="", help="opt_root")
parser.add_argument("", "--agg", type=int, default="10", help="agg")
parser.add_argument("", "--format", type=str, default="wav", help="format")
parser.add_argument("", "--is_half", type=str, default="", help="is_half")
parser.add_argument("", "--device", type=str, default="", help="device")
args = parser.parse_args()

inp_root=args.inp_root
opt_root=args.opt_root
model_name=args.model_name
format=args.format
device=args.device
agg=int(args.agg)
is_half = eval(args.is_half)

def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    try:
        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        is_hp3 = "HP3" in model_name
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=device,
                is_half=is_half,
            )
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]
        for path in paths:
            inp_path = os.path.join(inp_root, path)
            if(os.path.isfile(inp_path)==False):continue
            need_reformat = 1
            done = 0
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0,is_hp3
                    )
                    done = 1
            except:
                need_reformat = 1
                traceback.print_exc()

            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (
                    os.path.join(os.environ["TEMP"]),
                    os.path.basename(inp_path),
                )
                os.system(
                    "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y"
                    % (inp_path, tmp_path)
                )
                inp_path = tmp_path

            if done == 0:
                pre_fun._path_audio_(
                    inp_path, save_root_ins, save_root_vocal, format0,is_hp3
                )
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



uvr(model_name, inp_root, opt_root, [], opt_root, agg, format)