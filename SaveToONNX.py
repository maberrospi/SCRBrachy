import torch
from UNet import UNet

dummy_input = torch.randn(4, 1, 512, 512)
model = UNet(n_channels=1, n_classes=1)
model.load_state_dict(torch.load("models/checkpoint_epoch4.pth", map_location="cpu"))
model.eval()
input_name = ["CTimage"]
output_name = ["Mask"]

torch.onnx.export(
    model,
    dummy_input,
    "ModelUNet.onnx",
    verbose=False,
    input_names=input_name,
    output_names=output_name,
)
