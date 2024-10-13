import os
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

# Suppress TensorFlow from trying to use GPUs since no GPUs are available
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Limit TensorFlow's memory growth to avoid unnecessary memory consumption
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Initialize the FastAPI app
app = FastAPI()

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the trained model
model = tf.keras.models.load_model('fake_news_detector/fake_news_detection_model.keras')

# Load the tokenizer
with open('fake_news_detector/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# Preprocess function for text input
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if len(token) > 3]
    return ' '.join(tokens)


# Define the request body using Pydantic's BaseModel
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


# POST endpoint to classify news, requiring the API key
@app.post("/news/predict/")
async def predict(request: NewsRequest, api_key: str = Depends(get_api_key)):
    # Get the text from the request
    text = request.text

    # Preprocess the text
    clean_text = preprocess(text)

    # Tokenize and pad the text
    sequences = tokenizer.texts_to_sequences([clean_text])
    maxlen = 200
    padded_sequence = pad_sequences(sequences, maxlen=maxlen, padding='post')

    # Make a prediction
    prediction = model.predict(padded_sequence)[0][0]

    # Interpret the prediction
    result = "Real News" if prediction > 0.5 else "Fake News"

    return {"prediction": result}
