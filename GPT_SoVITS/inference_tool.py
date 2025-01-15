'''
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
'''
import os, logging
import argparse
import soundfile as sf
from feature_extractor import cnhubert

from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav


logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)


parser = argparse.ArgumentParser(description="GPT-SoVITS inference tool")
parser.add_argument("-gp", "--gpt_path", type=str, default="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt", help="gpt_path")
parser.add_argument("-sp", "--sovits_path", type=str, default="GPT_SoVITS/pretrained_models/s2G488k.pth", help="sovits_path")
parser.add_argument("-cp", "--cnhubert_base_path", type=str, default="GPT_SoVITS/pretrained_models/chinese-hubert-base", help="cnhubert_base_path")
parser.add_argument("-bp", "--bert_path", type=str, default="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large", help="bert_path")
parser.add_argument("-rfp", "--ref_wav_path", type=str, default="", help="ref_wav_path")
parser.add_argument("-pt", "--prompt_text", type=str, default="", help="prompt_text")
parser.add_argument("-pl", "--prompt_language", type=str, default="", help="prompt_language")
parser.add_argument("-t", "--text", type=str, default="", help="text")
parser.add_argument("-tl", "--text_language", type=str, default="", help="text_language")
parser.add_argument("-o", "--out_path", type=str, default="", help="out_path")
parser.add_argument("-hc", "--how_to_cut", type=str, default="chinese_period", help="how_to_cut")
parser.add_argument("-ih", "--is_half", type=str, default="", help="is_half")
parser.add_argument("-g", "--gpus", type=str, default="0", help="gpus")
parser.add_argument("-tk", "--top_k", type=str, default="10", help="top_k")
parser.add_argument("-tp", "--top_p", type=str, default="0.1", help="top_p")
parser.add_argument("-tmp", "--temperature", type=str, default="0.6", help="temperature")
args = parser.parse_args()

gpt_path=args.gpt_path
sovits_path=args.sovits_path
cnhubert.cnhubert_base_path=args.cnhubert_base_path
bert_path=args.bert_path
ref_wav_path=args.ref_wav_path
prompt_text=args.prompt_text
prompt_language=args.prompt_language
text=args.text
text_language=args.text_language
out_path = args.out_path
how_to_cut=args.how_to_cut
top_k=int(args.top_k)
top_p=float(args.top_p)
temperature=float(args.temperature)
is_half = eval(args.is_half)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

i18n = I18nAuto()


def synthesize(GPT_model_path, SoVITS_model_path, ref_audio_path, ref_text_path, ref_language, target_text_path,
               target_language, output_path, how_to_cut=i18n("不切"), top_k=20, top_p=0.6, temperature=0.6, ref_free=False):
    # Read reference text
    with open(ref_text_path, 'r', encoding='utf-8') as file:
        ref_text = file.read()

    # Read target text
    with open(target_text_path, 'r', encoding='utf-8') as file:
        target_text = file.read()

    # Change model weights
    change_gpt_weights(gpt_path=GPT_model_path)
    change_sovits_weights(sovits_path=SoVITS_model_path)

    # Synthesize audio
    synthesis_result = get_tts_wav(ref_wav_path=ref_audio_path,
                                   prompt_text=ref_text,
                                   prompt_language=i18n(ref_language),
                                   text=target_text,
                                   text_language=i18n(target_language),
                                   how_to_cut=how_to_cut,
                                   top_k=top_k,
                                   top_p=top_p,
                                   temperature=temperature,
                                   ref_free=ref_free
                                   )

    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        sf.write(output_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_path}")


synthesize(gpt_path, sovits_path, ref_wav_path, prompt_text, prompt_language, text, text_language, out_path, how_to_cut, top_k, top_p, temperature)