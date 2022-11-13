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

import Markdown
display(Markdown.md"""
> The culprit is probably *Expand_143*.

```
# after exp torch.Size([160]) torch.Size([1])

julia:
["timesteps"]::(64,) = Expand_143(["timestep", "onnx::Expand_703"])

python:
emb = timesteps[:, None].float() * emb[None, :]

# after exp mul torch.Size([1, 160])
```
""")
# The culpri

timestep = [1]
encoder_hidden_states = fill(Flux.nil, 64,64,4,1)
#                                      ^    ^
#                            encoder_dim    number_of_tokens

# path = "~/irisa/diffusers/decoder_v1_4_pytorch_1_1.onnx"
# path = joinpath(@__DIR__, "model_sim.onnx")
# path = "~/irisa/diffusers/decoder_v1_4_pytorch.onnx"
# path = "~/irisa/diffusers/decoder_v1_4_pytorch_sim.onnx"
path = "~/Projects/wnn/model.onnx"

# model = @time ONNX.load(expanduser(path), x, timestep, hidden_states; exec=true)
model = @time ONNX.load(expanduser(path), x, timestep, encoder_hidden_states; exec=true)

out = Umlaut.play!(model, x);
@show size(out) typeof(out)

Umlaut.inputs!(model, rand(Float32, latent_size...))
ONNX.save("./model_512.onnx", model)

