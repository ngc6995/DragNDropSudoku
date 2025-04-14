import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.onnx

class SudokuDigitCNN(nn.Module):
    def __init__(self):
        super(SudokuDigitCNN, self).__init__()
        # Conv Layer 1: More filters for better feature extraction
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Conv Layer 2: Deeper feature extraction
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Conv Layer 3: Additional depth
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=9)

    def forward(self, x):
        # Input shape: [batch_size, 1, 28, 28] (assumed)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14x14 -> 7x7
        x = F.relu(self.bn3(self.conv3(x)))             # 7x7 -> 7x7 (no pooling)
        x = torch.flatten(x, 1)                         # Flatten: [batch_size, 64*7*7]
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)                                 # Logits for 9 classes
        return x

def export_to_onnx(model, input_shape, onnx_file_path):
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input tensor with the specified shape
    dummy_input = torch.randn(input_shape)
    
    # Export the model to ONNX
    torch.onnx.export(
        model,                    # PyTorch model
        dummy_input,              # Example input tensor
        onnx_file_path,           # Output ONNX file path
        export_params=True,       # Store trained weights
        opset_version=11,         # ONNX opset version
        do_constant_folding=True, # Optimize by folding constants
        input_names=['input'],    # Input layer name
        output_names=['output'],  # Output layer name
        dynamic_axes={            # Support dynamic batch size
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {onnx_file_path}")

# Example usage
if __name__ == "__main__":
    # Instantiate the model
    model = SudokuDigitCNN()
    
    # Load the pretrained state dictionary
    state_dict = torch.load("digits_20250410195037_9984.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)  # Load weights into the model
    
    # Define input shape
    input_shape = (1, 1, 28, 28)  # batch_size, channels, height, width
    
    # Export to ONNX
    export_to_onnx(model, input_shape, "digits.onnx")
