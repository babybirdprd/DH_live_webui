// Global variables
let audioModel = null;
let renderModelRefInput = null;
let renderModelInterface = null;
let isModelLoaded = false;
let mediaRecorder = null;
let audioChunks = [];
let referenceVideo = null;
let referenceFeatures = null;

// DOM elements
const statusElement = document.getElementById('statusElement');
const referenceVideoInput = document.getElementById('referenceVideo');
const referenceVideoElement = document.getElementById('referenceVideoElement');
const processReferenceBtn = document.getElementById('processReferenceBtn');
const startRecordingBtn = document.getElementById('startRecordingBtn');
const stopRecordingBtn = document.getElementById('stopRecordingBtn');
const audioFileInput = document.getElementById('audioFile');
const generateBtn = document.getElementById('generateBtn');
const outputCanvas = document.getElementById('outputCanvas');
const ctx = outputCanvas.getContext('2d');

// Initialize the application
async function init() {
    updateStatus('Loading ONNX models...');
    
    try {
        // Set ONNX WebAssembly execution provider
        const ort = window.ort;
        ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
        
        // Load models
        audioModel = await ort.InferenceSession.create('./models/audio_model.onnx');
        renderModelRefInput = await ort.InferenceSession.create('./models/render_model_ref_input.onnx');
        renderModelInterface = await ort.InferenceSession.create('./models/render_model_interface.onnx');
        
        isModelLoaded = true;
        updateStatus('Models loaded successfully. Upload a reference video to begin.');
        
        // Enable buttons
        processReferenceBtn.disabled = false;
        
    } catch (error) {
        updateStatus(`Error loading models: ${error.message}`);
        console.error('Error loading models:', error);
    }
}

// Update status message
function updateStatus(message) {
    statusElement.textContent = `Status: ${message}`;
    console.log(message);
}

// Process reference video
async function processReferenceVideo(videoFile) {
    if (!isModelLoaded) {
        updateStatus('Models not loaded yet. Please wait.');
        return;
    }
    
    updateStatus('Processing reference video...');
    
    try {
        // Create video element and load the file
        const videoURL = URL.createObjectURL(videoFile);
        referenceVideoElement.src = videoURL;
        
        // Wait for video to load metadata
        await new Promise(resolve => {
            referenceVideoElement.onloadedmetadata = resolve;
        });
        
        // Extract frames from the video
        const frameCount = 5; // Number of reference frames to extract
        const frames = await extractFrames(referenceVideoElement, frameCount);
        
        // Process frames with the reference model
        referenceFeatures = await processReferenceFrames(frames);
        
        updateStatus('Reference video processed successfully.');
        generateBtn.disabled = false;
        
    } catch (error) {
        updateStatus(`Error processing reference video: ${error.message}`);
        console.error('Error processing reference video:', error);
    }
}

// Extract frames from video
async function extractFrames(videoElement, frameCount) {
    const frames = [];
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas dimensions to match video
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    
    // Calculate frame positions
    const duration = videoElement.duration;
    const interval = duration / (frameCount + 1);
    
    for (let i = 0; i < frameCount; i++) {
        const time = interval * (i + 1);
        videoElement.currentTime = time;
        
        // Wait for the video to seek to the specified time
        await new Promise(resolve => {
            videoElement.onseeked = resolve;
        });
        
        // Draw the video frame to the canvas
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        // Get the frame data
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        frames.push(imageData);
    }
    
    return frames;
}

// Process reference frames with the model
async function processReferenceFrames(frames) {
    // Combine frames into a single tensor
    const inputTensor = new Float32Array(frames.length * 3 * 256 * 256);
    
    // Resize and normalize frames
    for (let i = 0; i < frames.length; i++) {
        const resizedFrame = resizeImageData(frames[i], 256, 256);
        const normalizedFrame = normalizeImageData(resizedFrame);
        
        // Copy normalized frame data to input tensor
        const offset = i * 3 * 256 * 256;
        for (let j = 0; j < normalizedFrame.length; j++) {
            inputTensor[offset + j] = normalizedFrame[j];
        }
    }
    
    // Create ONNX tensor
    const tensor = new ort.Tensor('float32', inputTensor, [1, frames.length * 3, 256, 256]);
    
    // Run inference
    const results = await renderModelRefInput.run({ 'ref_img': tensor });
    
    return results;
}

// Resize image data to specified dimensions
function resizeImageData(imageData, width, height) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = width;
    canvas.height = height;
    
    // Create a temporary canvas to draw the original image data
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    tempCtx.putImageData(imageData, 0, 0);
    
    // Draw the resized image
    ctx.drawImage(tempCanvas, 0, 0, width, height);
    
    return ctx.getImageData(0, 0, width, height);
}

// Normalize image data to [0, 1] range
function normalizeImageData(imageData) {
    const data = imageData.data;
    const normalizedData = new Float32Array(3 * imageData.width * imageData.height);
    
    // Convert RGBA to RGB and normalize
    for (let i = 0; i < data.length / 4; i++) {
        normalizedData[i * 3] = data[i * 4] / 255.0;     // R
        normalizedData[i * 3 + 1] = data[i * 4 + 1] / 255.0; // G
        normalizedData[i * 3 + 2] = data[i * 4 + 2] / 255.0; // B
    }
    
    return normalizedData;
}

// Process audio data
async function processAudio(audioData) {
    if (!isModelLoaded || !referenceFeatures) {
        updateStatus('Models or reference video not loaded yet.');
        return;
    }
    
    updateStatus('Processing audio...');
    
    try {
        // Convert audio data to features
        const audioFeatures = await extractAudioFeatures(audioData);
        
        // Run audio model inference
        const audioResults = await runAudioModel(audioFeatures);
        
        // Generate animation frames
        const animationFrames = await generateAnimationFrames(audioResults);
        
        // Display animation
        displayAnimation(animationFrames);
        
        updateStatus('Animation generated successfully.');
        
    } catch (error) {
        updateStatus(`Error processing audio: ${error.message}`);
        console.error('Error processing audio:', error);
    }
}

// Extract audio features
async function extractAudioFeatures(audioData) {
    // This is a placeholder - in a real implementation, you would:
    // 1. Convert the audio data to the correct format (16kHz, mono)
    // 2. Extract the features using the same algorithm as in the Python code
    // 3. Return the features in the format expected by the model
    
    // For now, we'll return dummy data
    return new Float32Array(2 * 80).fill(0.1);
}

// Run audio model inference
async function runAudioModel(audioFeatures) {
    // Create input tensors
    const featuresTensor = new ort.Tensor('float32', audioFeatures, [1, 2, 80]);
    const h0Tensor = new ort.Tensor('float32', new Float32Array(2 * 1 * 192).fill(0), [2, 1, 192]);
    const c0Tensor = new ort.Tensor('float32', new Float32Array(2 * 1 * 192).fill(0), [2, 1, 192]);
    
    // Run inference
    const results = await audioModel.run({
        'audio_features': featuresTensor,
        'h0': h0Tensor,
        'c0': c0Tensor
    });
    
    return results;
}

// Generate animation frames
async function generateAnimationFrames(audioResults) {
    const frames = [];
    const pred = audioResults.pred;
    
    // For each prediction, generate a frame
    for (let i = 0; i < pred.dims[1]; i++) {
        // Create source image and prompt tensors
        const sourceImg = new ort.Tensor('float32', new Float32Array(3 * 256 * 256).fill(0.5), [1, 3, 256, 256]);
        const sourcePrompt = new ort.Tensor('float32', new Float32Array(3 * 256 * 256).fill(0.5), [1, 3, 256, 256]);
        
        // Run interface model
        const results = await renderModelInterface.run({
            'source_img': sourceImg,
            'source_prompt': sourcePrompt
        });
        
        frames.push(results.output);
    }
    
    return frames;
}

// Display animation on canvas
function displayAnimation(frames) {
    let frameIndex = 0;
    
    function renderFrame() {
        if (frameIndex >= frames.length) {
            frameIndex = 0;
        }
        
        const frame = frames[frameIndex];
        const imageData = new ImageData(
            new Uint8ClampedArray(frame.data.map(v => v * 255)),
            frame.dims[3],
            frame.dims[2]
        );
        
        ctx.putImageData(imageData, 0, 0);
        frameIndex++;
        
        requestAnimationFrame(renderFrame);
    }
    
    renderFrame();
}

// Event listeners
referenceVideoInput.addEventListener('change', (event) => {
    if (event.target.files.length > 0) {
        referenceVideo = event.target.files[0];
        const videoURL = URL.createObjectURL(referenceVideo);
        referenceVideoElement.src = videoURL;
        processReferenceBtn.disabled = false;
    }
});

processReferenceBtn.addEventListener('click', () => {
    if (referenceVideo) {
        processReferenceVideo(referenceVideo);
    }
});

startRecordingBtn.addEventListener('click', async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.addEventListener('dataavailable', (event) => {
            audioChunks.push(event.data);
        });
        
        mediaRecorder.addEventListener('stop', () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            processAudio(audioBlob);
        });
        
        mediaRecorder.start();
        startRecordingBtn.disabled = true;
        stopRecordingBtn.disabled = false;
        updateStatus('Recording audio...');
        
    } catch (error) {
        updateStatus(`Error starting recording: ${error.message}`);
        console.error('Error starting recording:', error);
    }
});

stopRecordingBtn.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        startRecordingBtn.disabled = false;
        stopRecordingBtn.disabled = true;
        updateStatus('Recording stopped. Processing audio...');
    }
});

audioFileInput.addEventListener('change', (event) => {
    if (event.target.files.length > 0) {
        const audioFile = event.target.files[0];
        processAudio(audioFile);
    }
});

generateBtn.addEventListener('click', () => {
    if (audioFileInput.files.length > 0) {
        processAudio(audioFileInput.files[0]);
    } else {
        updateStatus('Please upload an audio file or record audio first.');
    }
});

// Initialize the application
window.addEventListener('DOMContentLoaded', init);