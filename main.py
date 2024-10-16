import os
import logging
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize the FastAPI app
app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the trained model
model = tf.keras.models.load_model('fake_news_detector/fake_news_detection_model.keras')

# Load the tokenizer
with open('fake_news_detector/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Constants
MAX_SEQUENCE_LENGTH = 500
PREDICTION_THRESHOLD = 0.7


# Preprocess function for text input
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)


# Define the request body
class NewsRequest(BaseModel):
    text: str


# Load the API key from the environment variable
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"

# Create a dependency that checks for the API key
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")


@app.post("/news/predict/")
async def predict(request: NewsRequest, api_key: str = Depends(get_api_key)):
    try:
        text = request.text
        clean_text = preprocess(text)
        sequences = tokenizer.texts_to_sequences([clean_text])
        padded_sequence = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

        prediction = model.predict(padded_sequence)[0][0]

        if prediction > PREDICTION_THRESHOLD:
            result = "Real News"
            confidence = round(prediction * 100, 2)
        elif prediction < (1 - PREDICTION_THRESHOLD):
            result = "Fake News"
            confidence = round((1 - prediction) * 100, 2)
        else:
            result = "Uncertain"
            confidence = round(abs(0.5 - prediction) * 200, 2)

        response = {
            "prediction": result,
            "confidence": confidence,
            "raw_score": float(prediction)
        }

        logger.info(f"Prediction made: {response}")
        return response

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
