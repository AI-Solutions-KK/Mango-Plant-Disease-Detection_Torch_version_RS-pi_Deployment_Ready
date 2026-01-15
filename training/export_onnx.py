import torch
import timm

# Load the SAME model used during training
model = timm.create_model(
    'efficientnet_b4',
    pretrained=True,
    num_classes=0
)
model.eval()

# Dummy input
dummy = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy,
    "efficientnet_b4.onnx",
    input_names=["input"],
    output_names=["features"],
    opset_version=11
)

print("ONNX created successfully")
