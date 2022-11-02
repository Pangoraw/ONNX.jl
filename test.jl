using Revise
import ONNX
import ONNX.Umlaut
# using CUDA
# using BenchmarkTools
using Flux

Base.cos(::Flux.Nil) = Flux.nil
Base.sin(::Flux.Nil) = Flux.nil

s = 64
latent_size = (s, s, 4, 1)

x = fill(Flux.nil, latent_size...)
# x = randn(Float32, latent_size...)

timestep = [Flux.nil]
hidden_states = fill(Flux.nil, latent_size...)

# path = "~/irisa/diffusers/decoder_v1_4_pytorch_1_1.onnx"
# path = joinpath(@__DIR__, "model_sim.onnx")
path = "~/Projects/wnn/unet_sim2.onnx"
# path = "~/irisa/diffusers/decoder_v1_4_pytorch.onnx"
# path = "~/irisa/diffusers/decoder_v1_4_pytorch_sim.onnx"

# model = @time ONNX.load(expanduser(path), x, timestep, hidden_states; exec=true)
model = @time ONNX.load(expanduser(path), x, timestep, hidden_states; exec=true)

out = Umlaut.play!(model, x);
@show size(out) typeof(out)

Umlaut.inputs!(model, rand(Float32, latent_size...))
ONNX.save("./model_512.onnx", model)
