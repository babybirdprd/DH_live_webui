import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import os

# Create a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create directory for ONNX models
os.makedirs("onnx_models", exist_ok=True)

# Initialize the model
model = SimpleModel()
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 10)

# Export the model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "onnx_models/simple_model.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Verify the model
onnx_model = onnx.load("onnx_models/simple_model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model verified successfully!")

# Test inference with ONNX Runtime
session = ort.InferenceSession("onnx_models/simple_model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Create random input data
input_data = np.random.randn(1, 10).astype(np.float32)

# Run inference
outputs = session.run([output_name], {input_name: input_data})
print(f"ONNX Runtime output shape: {outputs[0].shape}")

print("ONNX conversion and inference test completed successfully!")

# Create a simple web page to test ONNX Runtime Web
os.makedirs("web/js", exist_ok=True)
os.makedirs("web/models", exist_ok=True)

# Copy the ONNX model to the web directory
import shutil
shutil.copy("onnx_models/simple_model.onnx", "web/models/")

# Create a simple HTML file
html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ONNX Runtime Web Test</title>
</head>
<body>
    <h1>ONNX Runtime Web Test</h1>
    <div id="result">Loading model...</div>
    
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
    <script>
        async function runModel() {
            try {
                // Set ONNX WebAssembly execution provider
                const ort = window.ort;
                ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
                
                // Load model
                const session = await ort.InferenceSession.create('./models/simple_model.onnx');
                
                // Create input tensor
                const inputData = new Float32Array(10).fill(0.1);
                const inputTensor = new ort.Tensor('float32', inputData, [1, 10]);
                
                // Run inference
                const results = await session.run({ 'input': inputTensor });
                const output = results.output;
                
                document.getElementById('result').innerHTML = 
                    `Model loaded and executed successfully!<br>
                     Output shape: ${output.dims.join('x')}<br>
                     Output data: ${Array.from(output.data).map(x => x.toFixed(4)).join(', ')}`;
            } catch (error) {
                document.getElementById('result').innerHTML = `Error: ${error.message}`;
                console.error('Error running model:', error);
            }
        }
        
        window.addEventListener('DOMContentLoaded', runModel);
    </script>
</body>
</html>
"""

with open("web/index.html", "w") as f:
    f.write(html_content)

print("Web test files created successfully!")
print("To test in a browser, start a web server with:")
print("python -m http.server 12000 --directory web")