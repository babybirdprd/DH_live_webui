# Digital Human ONNX Conversion and Web Implementation

This extension to the Digital Human project enables running the inference models in web browsers using ONNX and WebAssembly.

## Overview

The implementation consists of:

1. **ONNX Model Conversion**: Converting PyTorch models to ONNX format
2. **Web Interface**: A JavaScript implementation for browser-based inference
3. **WebAssembly Integration**: Using ONNX Runtime Web for efficient inference
4. **Server-Side API**: Flask server for handling API requests and serving the web application

## Getting Started

### 1. Install Dependencies

First, install the required Python packages:

```bash
pip install torch onnx onnxruntime flask flask_cors opencv-python kaldi_native_fbank
```

### 2. Convert Models to ONNX

Make sure you have the PyTorch models available in the `checkpoint` directory:
- `checkpoint/audio.pkl`: The audio processing model
- `checkpoint/render.pth`: The rendering model

Then run the conversion script:

```bash
python convert_to_onnx.py
```

This will create ONNX versions of the models in the `onnx_models` directory:
- `audio_model.onnx`: The audio processing model
- `render_model_ref_input.onnx`: The reference input part of the render model
- `render_model_interface.onnx`: The interface part of the render model

### 3. Start the Web Server

Run the web server to serve the web application and handle API requests:

```bash
python web_server.py --port 12000 --host 0.0.0.0
```

Add the `--convert` flag to automatically convert models to ONNX format:

```bash
python web_server.py --port 12000 --host 0.0.0.0 --convert
```

### 4. Access the Web Interface

Open a web browser and navigate to:

```
http://localhost:12000
```

## Implementation Details

### ONNX Conversion

The conversion process splits the models into separate components:

1. **Audio Model**: Converts audio input to mouth movement parameters
   - Inputs: `audio_features` (shape: [batch_size, sequence_length, feature_dim]), `h0` and `c0` (LSTM initial states)
   - Outputs: Mouth movement parameters

2. **Render Model - Reference Input**: Processes reference video frames
   - Inputs: `ref_img` (shape: [batch_size, num_frames, height, width])
   - Outputs: Reference features for animation

3. **Render Model - Interface**: Generates animated frames based on audio features
   - Inputs: `source_img` and `source_prompt` (shape: [batch_size, 3, height, width])
   - Outputs: Generated animation frames

### Model Architecture

The Digital Human system consists of two main components:

1. **Audio Model**: An LSTM-based network that processes audio features and generates mouth movement parameters.
2. **Render Model**: A complex network with two main functions:
   - `ref_input`: Processes reference video frames to extract appearance features
   - `interface`: Generates animated frames by combining source image with audio-driven mouth movements

### Web Implementation

The web implementation uses:

- **ONNX Runtime Web**: For running the models in the browser
- **WebAssembly**: For efficient execution
- **Canvas API**: For rendering the output frames
- **MediaRecorder API**: For capturing audio input

The web interface provides:
- Reference video upload and processing
- Audio recording or file upload
- Real-time animation display

### Server-Side Implementation

The server-side implementation uses:
- **Flask**: For handling HTTP requests and serving the web application
- **ONNX Runtime**: For running inference on the server
- **API Endpoints**:
  - `/api/process_audio`: Processes audio input and returns mouth movement parameters
  - `/api/process_reference`: Processes reference video and extracts features

## Optimization Techniques

Several optimization techniques are used to improve performance:

1. **Model Splitting**: The render model is split into two parts to optimize memory usage
2. **Dynamic Axes**: ONNX models are exported with dynamic batch size for flexibility
3. **WebAssembly Multithreading**: ONNX Runtime Web is configured to use multiple threads

## Limitations

The current web implementation has some limitations:

1. **Performance**: Complex operations may be slower in the browser compared to native execution
2. **Memory Usage**: Large models may consume significant browser memory
3. **Feature Parity**: Some advanced features of the Python implementation may not be available
4. **Browser Compatibility**: WebAssembly and ONNX Runtime Web require modern browsers

## Future Improvements

Potential improvements include:

1. **Model Optimization**: Quantizing models for smaller size and faster inference
2. **WebGL Integration**: Using WebGL for faster rendering
3. **Progressive Loading**: Loading models progressively to improve startup time
4. **WebRTC Integration**: Adding real-time communication capabilities
5. **Model Compression**: Applying techniques like pruning and quantization to reduce model size
6. **Mobile Optimization**: Adapting the implementation for mobile devices

## Troubleshooting

If you encounter issues:

1. **Check Browser Console**: Most errors will be logged to the browser's developer console
2. **WebAssembly Support**: Ensure your browser supports WebAssembly
3. **Memory Limits**: If the application crashes, your browser may need more memory allocation
4. **CORS Issues**: Make sure the server is properly configured to allow cross-origin requests
5. **Model Loading Errors**: Check that the ONNX models are correctly exported and accessible
6. **API Errors**: Check the server logs for detailed error messages

## References

- [ONNX Runtime Web Documentation](https://onnxruntime.ai/docs/tutorials/web/)
- [WebAssembly Documentation](https://webassembly.org/docs/high-level-goals/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [PyTorch to ONNX Conversion Guide](https://pytorch.org/docs/stable/onnx.html)