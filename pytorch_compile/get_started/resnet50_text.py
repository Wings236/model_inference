import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# print(model)
model = model.cuda()
opt_model = torch.compile(model, backend="inductor")
a = opt_model(torch.randn(1,3,64,64).cuda())
# a = model(torch.randn(1,3,64,64).cuda(1))
print(a)
