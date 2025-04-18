import os
import sys
import argparse
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import numpy as np
import onnx
import onnxruntime as ort

app = Flask(__name__)
CORS(app)

# Create directories
os.makedirs("web/models", exist_ok=True)

# Global variables
audio_session = None
render_ref_session = None
render_interface_session = None

@app.route('/')
def index():
    return send_from_directory('web', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)

@app.route('/models/<path:path>')
def serve_models(path):
    return send_from_directory('web/models', path)

@app.route('/api/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    audio_path = os.path.join('temp', 'audio.wav')
    os.makedirs('temp', exist_ok=True)
    audio_file.save(audio_path)
    
    try:
        # Extract audio features (placeholder - in a real implementation, you would use the same
        # feature extraction as in the Python code)
        audio_features = np.random.rand(1, 10, 80).astype(np.float32)  # Example shape
        
        # Create input tensors
        h0 = np.zeros((2, 1, 192), dtype=np.float32)
        c0 = np.zeros((2, 1, 192), dtype=np.float32)
        
        # Run inference with ONNX Runtime
        inputs = {
            'audio_features': audio_features,
            'h0': h0,
            'c0': c0
        }
        
        mouth_frames = audio_session.run(None, inputs)[0]  # Get the first output
        
        # Convert to list for JSON serialization
        frames_list = mouth_frames.tolist()
        
        return jsonify({'frames': frames_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_reference', methods=['POST'])
def process_reference():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    video_path = os.path.join('temp', 'reference.mp4')
    os.makedirs('temp', exist_ok=True)
    video_file.save(video_path)
    
    try:
        # Extract frames from video (placeholder)
        ref_img = np.random.rand(1, 30, 256, 256).astype(np.float32)
        
        # Run inference with ONNX Runtime
        inputs = {'ref_img': ref_img}
        ref_features = render_ref_session.run(None, inputs)[0]
        
        # Store the reference features in a session variable
        # In a real implementation, you would store this in a database or session
        request.environ['ref_features'] = ref_features
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def convert_models_to_onnx():
    """Convert PyTorch models to ONNX format"""
    from convert_to_onnx import convert_audio_model_to_onnx, convert_render_model_to_onnx
    
    try:
        # Create checkpoint directory if it doesn't exist
        os.makedirs("checkpoint", exist_ok=True)
        os.makedirs("onnx_models", exist_ok=True)
        
        # Check if models exist
        if not os.path.exists("checkpoint/audio.pkl") or not os.path.exists("checkpoint/render.pth"):
            print("Model files not found. Please make sure the models are available in the checkpoint directory.")
            return False
        
        # Convert models to ONNX
        convert_audio_model_to_onnx()
        convert_render_model_to_onnx()
        
        # Copy ONNX models to web directory
        os.system("cp onnx_models/*.onnx web/models/")
        
        return True
    except Exception as e:
        print(f"Error converting models to ONNX: {e}")
        return False

def initialize_onnx_sessions():
    """Initialize ONNX Runtime sessions for inference"""
    global audio_session, render_ref_session, render_interface_session
    
    try:
        # Check if ONNX models exist
        if not os.path.exists("onnx_models/audio_model.onnx") or not os.path.exists("onnx_models/render_model_ref_input.onnx"):
            print("ONNX model files not found. Please run with --convert flag first.")
            return False
        
        # Initialize ONNX Runtime sessions
        audio_session = ort.InferenceSession("onnx_models/audio_model.onnx")
        render_ref_session = ort.InferenceSession("onnx_models/render_model_ref_input.onnx")
        
        # Check if render_model_interface.onnx exists
        if os.path.exists("onnx_models/render_model_interface.onnx"):
            render_interface_session = ort.InferenceSession("onnx_models/render_model_interface.onnx")
        
        return True
    except Exception as e:
        print(f"Error initializing ONNX sessions: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Digital Human Web Server')
    parser.add_argument('--port', type=int, default=12000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--convert', action='store_true', help='Convert models to ONNX format')
    args = parser.parse_args()
    
    if args.convert:
        print("Converting models to ONNX format...")
        if convert_models_to_onnx():
            print("Models converted successfully!")
        else:
            print("Failed to convert models to ONNX format.")
            sys.exit(1)
    
    print("Initializing ONNX sessions...")
    if initialize_onnx_sessions():
        print("ONNX sessions initialized successfully!")
    else:
        print("Failed to initialize ONNX sessions.")
        sys.exit(1)
    
    print(f"Starting server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)