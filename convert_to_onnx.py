import os
import torch
import numpy as np
import onnx
import onnxruntime as ort
from talkingface.models.audio2bs_lstm import Audio2Feature
from talkingface.models.DINet import DINet_five_Ref

# Create checkpoint directory if it doesn't exist
os.makedirs("checkpoint", exist_ok=True)
os.makedirs("onnx_models", exist_ok=True)

# Check if models exist, if not, create dummy models for testing
if not os.path.exists("checkpoint/audio.pkl"):
    print("Creating dummy audio model for testing...")
    audio_model = Audio2Feature()
    torch.save(audio_model.state_dict(), "checkpoint/audio.pkl")

if not os.path.exists("checkpoint/render.pth"):
    print("Creating dummy render model for testing...")
    render_model = DINet_five_Ref(6, 30)
    torch.save(render_model.state_dict(), "checkpoint/render.pth")

# Convert Audio Model to ONNX
def convert_audio_model_to_onnx():
    print("Converting Audio Model to ONNX...")
    model = Audio2Feature()
    model.load_state_dict(torch.load("checkpoint/audio.pkl"))
    model.eval()
    
    # Create dummy inputs
    audio_features = torch.randn(1, 2, 80)  # [batch_size, seq_len, ndim]
    h0 = torch.zeros(2, 1, 192)
    c0 = torch.zeros(2, 1, 192)
    
    # Export the model
    torch.onnx.export(
        model,
        (audio_features, h0, c0),
        "onnx_models/audio_model.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['audio_features', 'h0', 'c0'],
        output_names=['pred', 'hn', 'cn'],
        dynamic_axes={
            'audio_features': {0: 'batch_size', 1: 'seq_len'},
            'h0': {1: 'batch_size'},
            'c0': {1: 'batch_size'},
            'pred': {0: 'batch_size', 1: 'seq_len'},
            'hn': {1: 'batch_size'},
            'cn': {1: 'batch_size'}
        }
    )
    
    # Verify the model
    onnx_model = onnx.load("onnx_models/audio_model.onnx")
    onnx.checker.check_model(onnx_model)
    print("Audio Model converted and verified successfully!")

# Convert Render Model to ONNX
def convert_render_model_to_onnx():
    print("Converting Render Model to ONNX...")
    model = DINet_five_Ref(6, 30)
    model.load_state_dict(torch.load("checkpoint/render.pth"))
    model.eval()
    
    # Create dummy inputs
    ref_img = torch.randn(1, 30, 256, 256)
    source_img = torch.randn(1, 3, 256, 256)
    source_prompt = torch.randn(1, 3, 256, 256)
    
    # Create a wrapper for ref_input
    class RefInputWrapper(torch.nn.Module):
        def __init__(self, model):
            super(RefInputWrapper, self).__init__()
            self.model = model
            
        def forward(self, ref_img):
            self.model.ref_input(ref_img)
            return self.model.ref_trans_feature0
    
    # Create a wrapper for interface
    class InterfaceWrapper(torch.nn.Module):
        def __init__(self, model):
            super(InterfaceWrapper, self).__init__()
            self.model = model
            
        def forward(self, source_img, source_prompt):
            return self.model.interface(source_img, source_prompt)
    
    # Initialize the model with reference input
    model.ref_input(ref_img)
    
    # Create wrappers
    ref_input_wrapper = RefInputWrapper(model)
    interface_wrapper = InterfaceWrapper(model)
    
    # Export ref_input function
    torch.onnx.export(
        ref_input_wrapper,
        (ref_img,),
        "onnx_models/render_model_ref_input.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['ref_img'],
        output_names=['ref_trans_feature0'],
        dynamic_axes={
            'ref_img': {0: 'batch_size'},
            'ref_trans_feature0': {0: 'batch_size'}
        }
    )
    
    # Export interface function
    torch.onnx.export(
        interface_wrapper,
        (source_img, source_prompt),
        "onnx_models/render_model_interface.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['source_img', 'source_prompt'],
        output_names=['output'],
        dynamic_axes={
            'source_img': {0: 'batch_size'},
            'source_prompt': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify the models
    ref_input_model = onnx.load("onnx_models/render_model_ref_input.onnx")
    onnx.checker.check_model(ref_input_model)
    
    interface_model = onnx.load("onnx_models/render_model_interface.onnx")
    onnx.checker.check_model(interface_model)
    
    print("Render Model converted and verified successfully!")

# Test ONNX models
def test_onnx_models():
    print("Testing ONNX models...")
    
    # Test Audio Model
    audio_session = ort.InferenceSession("onnx_models/audio_model.onnx")
    audio_input_name = audio_session.get_inputs()[0].name
    h0_input_name = audio_session.get_inputs()[1].name
    c0_input_name = audio_session.get_inputs()[2].name
    
    audio_features = np.random.randn(1, 2, 80).astype(np.float32)
    h0 = np.zeros((2, 1, 192), dtype=np.float32)
    c0 = np.zeros((2, 1, 192), dtype=np.float32)
    
    audio_outputs = audio_session.run(None, {
        audio_input_name: audio_features,
        h0_input_name: h0,
        c0_input_name: c0
    })
    
    print(f"Audio Model Output Shapes: {[output.shape for output in audio_outputs]}")
    
    # Test Render Model - ref_input
    ref_input_session = ort.InferenceSession("onnx_models/render_model_ref_input.onnx")
    ref_input_name = ref_input_session.get_inputs()[0].name
    
    ref_img = np.random.randn(1, 30, 256, 256).astype(np.float32)
    ref_input_outputs = ref_input_session.run(None, {ref_input_name: ref_img})
    
    # Test Render Model - interface
    interface_session = ort.InferenceSession("onnx_models/render_model_interface.onnx")
    source_img_name = interface_session.get_inputs()[0].name
    source_prompt_name = interface_session.get_inputs()[1].name
    
    source_img = np.random.randn(1, 3, 256, 256).astype(np.float32)
    source_prompt = np.random.randn(1, 3, 256, 256).astype(np.float32)
    
    interface_outputs = interface_session.run(None, {
        source_img_name: source_img,
        source_prompt_name: source_prompt
    })
    
    print(f"Render Model Interface Output Shape: {interface_outputs[0].shape}")
    
    print("ONNX models tested successfully!")

if __name__ == "__main__":
    try:
        convert_audio_model_to_onnx()
        convert_render_model_to_onnx()
        test_onnx_models()
        print("All models converted to ONNX successfully!")
    except Exception as e:
        print(f"Error converting models to ONNX: {e}")