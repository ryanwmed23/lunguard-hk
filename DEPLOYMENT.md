# Deployment Guide for LungGuard HK

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model** (First time only)
   ```bash
   python voc_classifier.py
   ```
   This creates `voc_model.pkl` and `voc_scaler.pkl` files.

3. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

## Deployment Options

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file path: `app.py`
5. Deploy!

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train model on startup (or pre-train and copy model files)
RUN python voc_classifier.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Local Production

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python voc_classifier.py

# Run with custom port
streamlit run app.py --server.port 8501
```

## Required Files

- `app.py` - Main Streamlit application
- `voc_classifier.py` - Model training and prediction functions
- `voc_model.pkl` - Trained model (generated after training)
- `voc_scaler.pkl` - Feature scaler (generated after training)
- `requirements.txt` - Python dependencies

## Environment Variables (Optional)

No environment variables required for basic deployment.

## Notes

- The model must be trained before the app can make predictions
- CSV upload expects columns matching VOC names (case-insensitive matching supported)
- PDF reports are generated on-demand
