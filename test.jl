using Revise
import ONNX

x = randn(64,64,4,1)
model = ONNX.load(expanduser("~/irisa/diffusers/decoder_v1_4_fp16_pytorch_sim.onnx"), x; exec=false)
model
