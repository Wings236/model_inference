import torch

@torch.jit.script  # 使用装饰器包装函数
def loop_function(x, y):
    for i in range(x.size(0)):  # 循环次数由输入x的尺寸决定
        x = x + y
    return x

class MyModel(torch.nn.Module):
    def forward(self, x, y):
        x = x - y
        x = loop_function(x, y)  # 调用脚本化的函数
        return x

model = MyModel()
scripted_model = torch.jit.script(model)  # 将整个模型脚本化

model = MyModel()
dummy_input = (torch.randn(2, 16, 32), torch.randn(2, 16, 32))

# 将模型转换为脚本模式
scripted_model = torch.jit.script(model)

# 导出ONNX模型 - 使用脚本化后的模型，并且不再需要`example_outputs`参数
torch.onnx.export(
    scripted_model,       # 使用脚本化后的模型
    dummy_input,
    "model_with_loop.onnx",
    input_names=['input1', 'input2'],
    output_names=['output'],
    dynamic_axes={
        'input1': {0: 'batch_size'},
        'input2': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=14      # 确保使用足够高的ONNX opset版本（例如12或更高）
)