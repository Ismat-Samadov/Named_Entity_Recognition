from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import numpy as np

app = FastAPI()

# Serve static files like CSS and JavaScript
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load the Hugging Face model and tokenizer
model_name = "IsmatS/xlm-roberta-az-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

label_mapping = {
    "LABEL_0": "Other",
    "LABEL_1": "Person",
    "LABEL_2": "Location",
    "LABEL_3": "Organization",
    "LABEL_4": "Date",
    "LABEL_5": "Time",
    "LABEL_6": "Money",
    "LABEL_7": "Percentage",
    "LABEL_8": "Facility",
    "LABEL_9": "Product",
    "LABEL_10": "Event",
    "LABEL_11": "Art",
    "LABEL_12": "Law",
    "LABEL_13": "Language",
    "LABEL_14": "Government",
    "LABEL_15": "Nationality or Religion",
    "LABEL_16": "Ordinal",
    "LABEL_17": "Cardinal",
    "LABEL_18": "Disease",
    "LABEL_19": "Contact",
    "LABEL_20": "Proverb or Saying",
    "LABEL_21": "Quantity",
    "LABEL_22": "Miscellaneous",
    "LABEL_23": "Position",
    "LABEL_24": "Project"
}

def convert_numpy_types(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_ner(text: str = Form(...)):
    ner_results = nlp_ner(text)
    
    # Initialize dictionary to store entities by type
    entities_by_type = {}

    # Process each detected entity
    for entity in ner_results:
        # Get the human-readable label
        entity_type = label_mapping.get(entity["entity_group"], entity["entity_group"])
        
        # Filter out non-entities (label "Other" in this case)
        if entity_type == "Other":
            continue
        
        # Add entity to the dictionary by its type
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []  # Initialize list for new entity type
        
        # Append the entity word to the corresponding type list
        entities_by_type[entity_type].append(entity["word"])

    return {"entities": entities_by_type}


# Run with uvicorn main:app --reload
# curl -X POST "http://127.0.0.1:8000/predict/" \
# -H "Content-Type: application/json" \
# -d '{"text": "Bakı şəhərində Azərbaycan Respublikasının prezidenti İlham Əliyev."}'

# 2014 - cu ilde Azərbaycan Respublikasının prezidenti İlham Əliyev Salyanda olub.