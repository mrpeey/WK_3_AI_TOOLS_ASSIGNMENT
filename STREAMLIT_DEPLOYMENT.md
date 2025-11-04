# MNIST Streamlit App - Deployment Guide

## ⚠️ Important: Streamlit Cloud Limitations

**PyTorch is too large (~700MB+) for Streamlit Cloud's free tier resource limits.**

## Deployment Options

### Option 1: Local Deployment (Recommended)

The app works perfectly locally with PyTorch installed:

```powershell
# Install requirements
pip install torch torchvision streamlit streamlit-drawable-canvas opencv-python pillow matplotlib numpy

# Run the app
streamlit run mnist_streamlit_app.py
```

The app will be available at `http://localhost:8501`

### Option 2: Use Pre-trained Model on Cloud (Alternative)

If you need cloud deployment, consider:

1. **Convert to ONNX format** (smaller ~5MB model file):
   - Requires C++ compiler which may not be available
   - ONNX Runtime is much smaller (~50MB vs ~700MB for PyTorch)

2. **Use TensorFlow/Keras** instead:
   - TensorFlow Lite models are very small
   - Better suited for cloud deployment

3. **Deploy on platforms with more resources**:
   - Heroku (paid tiers)
   - Google Cloud Run
   - AWS Elastic Beanstalk
   - Your own server

## App Features

- ✏️ Interactive canvas for drawing digits
- 🎯 Real-time CNN predictions
- 📊 Confidence scores and probability distributions  
- 🎨 Adjustable stroke width and drawing modes
- 📈 Model performance metrics

## Repository Contents

- `mnist_streamlit_app.py` - Main Streamlit application
- `best_mnist_cnn.pth` - Trained PyTorch model (~5MB)
- `requirements.txt` - Python dependencies
- `convert_to_onnx.py` - Script to convert model to ONNX (if needed)

## Local Development

The app runs flawlessly locally and provides an excellent interactive experience for digit recognition with the trained CNN model achieving >98% accuracy.
