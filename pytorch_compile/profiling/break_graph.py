import torch
import torch._dynamo
# user can switch between cuda and xpu
device = 'cuda'

class ModelWithBreaks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        def create_sequential():
            return torch.nn.Sequential(
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
            )
        self.mod1 = create_sequential()
        self.mod2 = create_sequential()
        self.mod3 = create_sequential()
        self.mod4 = create_sequential()

    def forward(self, inp):
        mod1 = self.mod1(inp)
        torch._dynamo.graph_break()
        mod2 = self.mod2(mod1)
        torch._dynamo.graph_break()
        mod3 = self.mod3(mod2)
        torch._dynamo.graph_break()
        mod4 = self.mod4(mod3)
        return mod4

model = ModelWithBreaks().to(device)
inputs = [torch.randn((128, 128), device=device) for _ in range(10)]

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

prof.export_chrome_trace("trace_break.json")