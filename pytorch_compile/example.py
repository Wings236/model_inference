import torch
import time
def fn(x):
    a = torch.cos(x)
    b = torch.sin(a)
    return b

new_fn = torch.compile(fn, backend="inductor")
input_tensor = torch.randn(100).to(device="cuda:1")
a = new_fn(input_tensor)

t1 = time.time()
x1 = fn(input_tensor)
t2 = time.time()
print(f"original time:{t2-t1}")
print(x1.shape)

t1 = time.time()
x2 = new_fn(input_tensor)
t2 = time.time()
print(f"compiled time:{t2-t1}")

print(x2.shape)