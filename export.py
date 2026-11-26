import sys,os,glob,torch,numpy as np
from huggingface_hub import snapshot_download
K=snapshot_download("Rabe3/kemetone")
code=snapshot_download("oddadmix/Nabra-7M-Distill")
sys.path[:0]=[code+"/kokoro_patched",code]
from kokoro import KModel
from kokoro.model import KModelForONNX
m=KModel(repo_id="oddadmix/Nabra-7M-Distill",config=K+"/config.json",model=K+"/kemetone.pth",disable_complex=True).eval()
w=KModelForONNX(m).eval()
T=32;ids=torch.randint(1,100,(1,T),dtype=torch.long);ids[0,0]=ids[0,-1]=0
os.makedirs("out",exist_ok=True)
torch.onnx.export(w,(ids,torch.randn(1,256),torch.tensor([1.0])),"out/kt_fp32.onnx",input_names=["input_ids","style","speed"],output_names=["waveform","durations"],dynamic_axes={"input_ids":{1:"T"},"waveform":{0:"N"},"durations":{0:"T"}},opset_version=17,do_constant_folding=True,dynamo=False)
v=torch.load(K+"/voices/kemetone.pt").reshape(-1,256).float().numpy();v.astype("<f4").tofile("out/voice.bin")
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
onnx.save(convert_float_to_float16(onnx.load("out/kt_fp32.onnx"),keep_io_types=True),"out/kt_fp16.onnx")
try:
    from onnxruntime.quantization import quantize_dynamic,QuantType
    quantize_dynamic("out/kt_fp32.onnx","out/kt_int8.onnx",weight_type=QuantType.QInt8)
except Exception as e: print("int8 failed",e)
for f in sorted(glob.glob("out/*")): print(f,os.path.getsize(f)/1e6,"MB")
