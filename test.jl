using Revise
import ONNX
import ONNX.Umlaut
# using CUDA
# using BenchmarkTools
using Flux

Base.cos(::Flux.Nil) = Flux.nil
Base.sin(::Flux.Nil) = Flux.nil
ONNX.erf(::Flux.Nil) = Flux.nil

use_nils = true

s = 64
latent_size = (s, s, 4, 1)

if use_nils
    x = fill(Flux.nil, latent_size...)
else
    x = randn(Float32, latent_size...)
end

timestep = [1]
encoder_hidden_states_size = (768, 100, 1)
#                             ^    ^
#                   encoder_dim    number_of_tokens
if use_nils
    encoder_hidden_states = fill(Flux.nil, encoder_hidden_states_size...)
else
    encoder_hidden_states = zeros(Float32, encoder_hidden_states_size...)
end

# path = "~/irisa/diffusers/decoder_v1_4_pytorch_1_1.onnx"
# path = joinpath(@__DIR__, "model_sim.onnx")
path = "~/Projects/wnn/model_sim2.onnx"
# path = "~/irisa/diffusers/decoder_v1_4_pytorch.onnx"
# path = "~/irisa/diffusers/decoder_v1_4_pytorch_sim.onnx"
# path = "~/Projects/wnn/model.onnx"

# model = @time ONNX.load(expanduser(path), x, timestep, hidden_states; exec=true)
model = @time ONNX.load(expanduser(path), x, timestep, encoder_hidden_states; exec=true)

out = Umlaut.play!(model, x, timestep, encoder_hidden_states);
@show size(out) typeof(out)

Umlaut.inputs!(model,
               rand(Float32, latent_size...),
               rand(Float32, size(timestep)...),
               rand(Float32, size(encoder_hidden_states)...))
ONNX.save("./model_512.onnx", model; external=true)
