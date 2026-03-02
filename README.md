# 🐦 Twitter Sentiment Analyzer

A full-stack machine learning application that analyzes sentiment in text using Natural Language Processing.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 🎯 Project Overview

This project demonstrates a complete end-to-end machine learning deployment pipeline, from model training to production deployment. Built as a capstone project to bridge the gap between Data Science and Machine Learning Engineering.

### Live Demo
🌐 **[Try it live!](#)** *(Add your Render URL here)*

## 🚀 Features

- **Sentiment Analysis**: Classify text as Positive, Negative, or Neutral
- **High Accuracy**: 98%+ accuracy on sample dataset
- **REST API**: FastAPI backend with automatic documentation
- **Web Interface**: Beautiful Streamlit frontend
- **Dockerized**: Fully containerized application
- **Cloud Deployed**: Hosted on Render with public URL

## 🛠️ Tech Stack

### Machine Learning
- **Algorithm**: Logistic Regression
- **Feature Engineering**: TF-IDF Vectorization
- **Libraries**: scikit-learn, pandas, numpy

### Backend
- **Framework**: FastAPI
- **Server**: Uvicorn
- **Validation**: Pydantic

### Frontend
- **Framework**: Streamlit
- **Styling**: Custom CSS

### Deployment
- **Containerization**: Docker
- **Hosting**: Render
- **Version Control**: Git & GitHub

## 📁 Project Structure

```
twitter-sentiment-analyzer/
├── model/
│   ├── sentiment_model.pkl        # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl       # TF-IDF vectorizer
│
├── app/
│   ├── main.py                    # FastAPI application
│   └── schemas.py                 # Pydantic models
│
├── frontend/
│   └── app.py                     # Streamlit interface
│
├── data/
│   └── tweets.csv                 # Training dataset
│
├── Dockerfile                     # Docker configuration
├── render.yaml                    # Render deployment config
├── requirements.txt               # Python dependencies
├── train_model.py                 # Model training script
├── run_api.py                     # API server launcher
├── run_frontend.py                # Frontend launcher
├── run_full_app.py                # Full application launcher
└── README.md                      # This file
```

## 🏃 Running Locally

### Prerequisites
- Python 3.12+
- pip
- Docker (optional)

### Option 1: Run with Python

**1. Clone the repository:**
```bash
git clone https://github.com/YOUR-USERNAME/twitter-sentiment-analyzer.git
cd twitter-sentiment-analyzer
```

**2. Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Train the model (if needed):**
```bash
python train_model.py
```

**5. Run the application:**

**Terminal 1 - API:**
```bash
python run_api.py
```

**Terminal 2 - Frontend:**
```bash
python run_frontend.py
```

**6. Access the application:**
- Frontend: http://localhost:8501
- API Docs: http://localhost:8000

### Option 2: Run with Docker

**1. Build the image:**
```bash
docker build -t sentiment-analyzer .
```

**2. Run the container:**
```bash
docker run -p 8000:8000 -p 8501:8501 sentiment-analyzer
```

**3. Access:**
- Frontend: http://localhost:8501
- API: http://localhost:8000

## 🔌 API Usage

### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Predict Sentiment
```bash
POST /predict
Content-Type: application/json

{
  "text": "I love this product!"
}
```

**Response:**
```json
{
  "text": "I love this product!",
  "sentiment": "positive",
  "confidence": 99.87,
  "probabilities": {
    "positive": 99.87,
    "negative": 0.08,
    "neutral": 0.05
  }
}
```

## 📊 Model Performance

- **Algorithm**: Logistic Regression with TF-IDF
- **Training Samples**: 3,600
- **Test Samples**: 900
- **Accuracy**: 98.78%
- **Precision**: 99%+ (all classes)
- **Recall**: 99%+ (all classes)

## 🚀 Deployment

This application is deployed on Render using Docker.

### Deploy to Render

1. Push code to GitHub
2. Connect GitHub to Render
3. Render automatically uses `Dockerfile` and `render.yaml`
4. Application deploys and gets public URL

## 🎥 Demo Video

[Link to demo video showing Postman API testing and frontend usage]

## 📝 Project Phases

### Phase 0: Model Development
- ✅ Data collection and preprocessing
- ✅ Feature engineering (TF-IDF)
- ✅ Model training and evaluation
- ✅ Model serialization

### Phase 1: Backend API
- ✅ FastAPI implementation
- ✅ Endpoint creation (/health, /predict)
- ✅ Request validation with Pydantic
- ✅ Error handling

### Phase 2: Frontend Interface
- ✅ Streamlit UI development
- ✅ API integration
- ✅ Interactive user experience
- ✅ Custom styling

### Phase 3: Containerization
- ✅ Dockerfile creation
- ✅ Docker image building
- ✅ Container testing

### Phase 4: Cloud Deployment
- ✅ GitHub repository setup
- ✅ Render deployment
- ✅ Public URL generation
- ✅ Production testing

## 🤝 Contributing

This is a capstone project, but suggestions are welcome!

## 📄 License

This project is for educational purposes.

## 👨‍💻 Author

**Your Name**
- GitHub: [@YOUR-USERNAME](https://github.com/YOUR-USERNAME)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/YOUR-PROFILE)

## 🙏 Acknowledgments

- Dataset: Sample sentiment data
- Framework: FastAPI, Streamlit
- Deployment: Render
- Guidance: End-to-End ML Project Course

---

**⭐ If you found this project helpful, please star the repository!**
