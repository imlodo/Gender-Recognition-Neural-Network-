import torch
import torch.nn as nn

class GenderCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(GenderCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# neural network
network = torch.load("genderCNN.pth")
# Input to the model
x = (torch.rand(1, 3, 64, 64)).to("cuda")
# Export the model
torch.onnx.export(network, # model being run
                  x, # model input (or a tuple for multiple inputs)
                  "gender.onnx", # where to save the model (can be a file or file-like object)
                  export_params=True, # store the trained parameter weights inside the model file
                  opset_version=10, # the ONNX version to export the model to
                  do_constant_folding=True, # whether to execute constant folding for optimization
                  input_names=['X'], # the model's input names
                  output_names=['Y'], # the model's output names
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK #resolve this error: operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
                  )