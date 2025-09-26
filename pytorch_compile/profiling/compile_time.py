import torch
from torchvision.models import resnet18

# user can switch between cuda and xpu
device = 'cuda'
model = resnet18().to(device)
inputs = [torch.randn((5, 3, 224, 224), device=device) for _ in range(10)]

model_c = torch.compile(model)

def fwd_bwd(inp):
    out = model_c(inp)
    out.sum().backward()

def warmup_compile():
    def fn(x):
        return x.sin().relu()

    x = torch.rand((2, 2), device=device, requires_grad=True)
    fn_c = torch.compile(fn)
    out = fn_c(x)
    out.sum().backward()

with torch.profiler.profile() as prof:
    with torch.profiler.record_function("warmup compile"):
        warmup_compile()

    with torch.profiler.record_function("resnet18 compile"):
        fwd_bwd(inputs[0])

prof.export_chrome_trace("trace_compile.json")