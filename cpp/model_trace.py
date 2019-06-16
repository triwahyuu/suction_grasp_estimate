import torch
import torchvision

# An instance of your model.
print("load model...")
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
print("tracing model...")
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model.pt")
print("tracing done, model saved...")