import torch
# 1 TorchScript
def f1(x, y):
    if x.sum() < 0:
        return -y
    return y

def test_fns(fn1, fn2, args):
    out1 = fn1(*args)
    out2 = fn2(*args)
    return torch.allclose(out1, out2)

inp1 = torch.randn(5, 5)
inp2 = torch.randn(5, 5)

traced_f1 = torch.jit.trace(f1, (inp1, inp2))
print("traced 1, 1:", test_fns(f1, traced_f1, (inp1, inp2)))
print("traced 1, 1:", test_fns(f1, traced_f1, (-inp1, inp2)))

# FX
import traceback as tb
try:
    torch.fx.symbolic_trace(f1)
except:
    tb.print_exc()

## tracing
fx_f1 = torch.fx.symbolic_trace(f1, concrete_args={"x": inp1})
print("fx 1, 1:", test_fns(f1, fx_f1, (inp1, inp2)))
print("fx 1, 1:", test_fns(f1, fx_f1, (-inp1, inp2)))

# torch.compile
torch._dynamo.reset()

compile_f1 = torch.compile(f1)
print("compile 1, 1:", test_fns(f1, compile_f1, (inp1, inp2)))
print("compile 1, 1:", test_fns(f1, compile_f1, (-inp1, inp2)))
print("~" * 10)

# TorchScript type error
def f2(x, y):
    return x + y

inp1 = torch.randn(5, 5)
inp2 = 3

script_f2 = torch.jit.script(f2)
try:
    script_f2(inp1, inp2)
except:
    tb.print_exc()

# torch.compile can do
compile_f2 = torch.compile(f2)
print("compile 2:", test_fns(f2, compile_f2, (inp1, inp2)))
print("~" * 10)

# non-Pytorch function
import scipy
def f3(x):
    x = x * 2
    x = scipy.fft.dct(x.numpy())
    x = torch.from_numpy(x)
    x = x * 2
    return x

inp1 = torch.randn(5, 5)
inp2 = torch.randn(5, 5)
traced_f3 = torch.jit.trace(f3, (inp1,))
print("traced 3:", test_fns(f3, traced_f3, (inp2,)))

# TorchScrpit
try:
    torch.jit.script(f3)
except:
    tb.print_exc()

# FX Tracing
try:
    torch.fx.symbolic_trace(f3)
except:
    tb.print_exc()

# torch.compile
#! however compile not support
torch._dynamo.reset()
compile_f3 = torch.compile(f3)
print("compile 3:", test_fns(f3, compile_f3, (inp2,)))