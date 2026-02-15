from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline, M2M100Tokenizer, M2M100ForConditionalGeneration
import torch
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model

# ==============================
# GLOBAL VARIABLES (IMPORTANT)
# ==============================

chatbot = None
model = None
translation_model = None
translation_tokenizer = None
translation_device = None
keras_model = None

# ==============================
# FASTAPI INIT
# ==============================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# CONSTANTS
# ==============================

MODEL_PATH = "./chest_disease_model.pth"

MAX_LENGTH = 1000
MIN_LENGTH = 100
TEMPERATURE = 0.8
TOP_P = 0.95
REPETITION_PENALTY = 1.2
LENGTH_PENALTY = 1.0

# ==============================
# MODEL LOADERS
# ==============================

def load_keras_model():
    global keras_model
    try:
        keras_model = tf.keras.models.load_model("./my_model")
        print("Keras model loaded")
    except Exception as e:
        print("Keras model failed:", e)
        keras_model = None


def load_chatbot():
    try:
        print("Loading chatbot...")
        bot = pipeline(
            "text-generation",
            model="./fine_tuned_medical_chatbotd",
            tokenizer="./fine_tuned_medical_chatbotd",
            device=-1,
        )
        print("Chatbot loaded")
        return bot
    except Exception as e:
        print("Chatbot failed:", e)
        return None


def load_disease_model():
    try:
        print("Loading disease model...")
        state_dict = torch.load(MODEL_PATH, map_location="cpu")

        m = models.densenet121(pretrained=False)
        m.classifier = torch.nn.Linear(m.classifier.in_features, 2)

        m.load_state_dict(state_dict, strict=False)
        m.eval()

        print("Disease model loaded")
        return m
    except Exception as e:
        print("Disease model failed:", e)
        return None


def load_translation_model():
    try:
        print("Loading translation model...")
        name = "facebook/m2m100_418M"

        tokenizer = M2M100Tokenizer.from_pretrained(name)
        model = M2M100ForConditionalGeneration.from_pretrained(name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        print("Translation model loaded")
        return model, tokenizer, device

    except Exception as e:
        print("Translation failed:", e)
        return None, None, None


# ==============================
# STARTUP EVENT (CRITICAL FIX)
# ==============================

@app.on_event("startup")
async def startup_event():
    global chatbot, model, translation_model, translation_tokenizer, translation_device

    print("=== STARTUP: Loading models ===")

    load_keras_model()

    chatbot = load_chatbot()

    model = load_disease_model()

    translation_model, translation_tokenizer, translation_device = load_translation_model()

    print("=== STARTUP COMPLETE ===")


# ==============================
# REQUEST MODELS
# ==============================

class ChatRequest(BaseModel):
    message: str
    language: str = None


class TranslationRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str


# ==============================
# IMAGE TRANSFORM
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ==============================
# ROOT ENDPOINT
# ==============================

@app.get("/")
async def root():
    return {"message": "Medical Chatbot API is up and running!"}


# ==============================
# CHAT ENDPOINT
# ==============================

@app.post("/chat")
async def chat(request: ChatRequest):

    if chatbot is None:
        return {"error": "Chatbot not loaded"}

    try:

        language_map = {
            "english": "en",
            "spanish": "es",
            "hindi": "hi",
            "bengali": "bn",
            "tamil": "ta",
            "telugu": "te",
            "marathi": "mr",
            "urdu": "ur",
            "gujarati": "gu",
            "kannada": "kn",
            "malayalam": "ml",
            "punjabi": "pa",
            "nepali": "ne"
        }

        lang = language_map.get(
            request.language.lower(), "en"
        ) if request.language else "en"

        input_text = request.message

        if lang != "en":
            input_text = translate(input_text, lang, "en")

        instruction = (
            "DONOT MENTION ANY DOCTOR NAME. "
            "REPLY LIKE A MEDICAL PROFESSIONAL."
        )

        formatted = f"{instruction}\n{input_text}\nDoctor:"

        response = chatbot(
            formatted,
            max_length=MAX_LENGTH,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=chatbot.tokenizer.eos_token_id
        )

        text = response[0]["generated_text"]

        reply = text.split("Doctor:")[-1].strip()

        if lang != "en":
            reply = translate(reply, "en", lang)

        return {"reply": reply}

    except Exception as e:
        return {"error": str(e)}


# ==============================
# TRANSLATE
# ==============================

def translate(text, src, tgt):

    translation_tokenizer.src_lang = src

    inputs = translation_tokenizer(text, return_tensors="pt").to(translation_device)

    tokens = translation_model.generate(
        **inputs,
        forced_bos_token_id=translation_tokenizer.get_lang_id(tgt)
    )

    return translation_tokenizer.decode(tokens[0], skip_special_tokens=True)


@app.post("/translate")
async def translate_api(request: TranslationRequest):

    result = translate(
        request.text,
        request.src_lang,
        request.tgt_lang
    )

    return {"translated_text": result}


# ==============================
# CHEST DISEASE PREDICT
# ==============================

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    if model is None:
        return {"error": "Disease model not loaded"}

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()

    return {"prediction": pred}


# ==============================
# BRAIN PREDICT
# ==============================

def preprocess_image(image, size=(64, 64)):
    image = image.resize(size).convert("L")
    arr = img_to_array(image) / 255.0
    return np.expand_dims(arr, axis=0)


@app.post("/predictbrain")
async def predictbrain(file: UploadFile = File(...)):

    if keras_model is None:
        return {"error": "Keras model not loaded"}

    image = Image.open(io.BytesIO(await file.read()))

    img = preprocess_image(image)

    prediction = keras_model.predict(img)

    result = "Tumor detected" if prediction[0][0] > 0.5 else "No tumor detected"

    return {
        "prediction": result,
        "confidence": float(prediction[0][0])
    }
