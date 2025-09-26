import torch
from torchvision.models import resnet18

device = "cuda"
model = resnet18().to(device)

inputs = [torch.randn((5, 3, 224, 224), device=device) for _ in range(10)]

model_c = torch.compile(model)

def fwd_bwd(inp):
    out = model_c(inp)
    out.sum().backward()

# warm up
fwd_bwd(inputs[0])

with torch.profiler.profile() as prof:
    for i in range(1, 4):
        fwd_bwd(inputs[i])
        prof.step()

prof.export_chrome_trace("trace.json")
