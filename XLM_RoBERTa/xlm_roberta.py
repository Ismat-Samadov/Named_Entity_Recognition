# -*- coding: utf-8 -*-
"""XLM-RoBERTa.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EYYZa7dya2RjTZXHSJ4pzIOgzqR8lmSk
"""

!pip install transformers datasets seqeval huggingface_hub

from huggingface_hub import login
login(token="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")
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer, AutoModelForTokenClassification, get_linear_schedule_with_warmup, EarlyStoppingCallback
from torch.optim import AdamW
from torch.amp import GradScaler
import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import ast


import os
os.environ["WANDB_DISABLED"] = "true"

import warnings
warnings.filterwarnings("ignore")

dataset = load_dataset("LocalDoc/azerbaijani-ner-dataset")
print(dataset)

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def preprocess_example(example):
    try:
        # Convert string representations of lists into actual lists
        example["tokens"] = ast.literal_eval(example["tokens"])
        example["ner_tags"] = list(map(int, ast.literal_eval(example["ner_tags"])))
    except (ValueError, SyntaxError) as e:
        # Handle the error: skip or provide default values if data is malformed
        print(f"Skipping malformed example: {example['index']} due to error: {e}")
        example["tokens"] = []
        example["ner_tags"] = []
    return example

# Apply the preprocessing step
dataset = dataset.map(preprocess_example)

def tokenize_and_align_labels(example):
    # Tokenize the tokens in the single example
    tokenized_inputs = tokenizer(
        example["tokens"],  # Pass tokens directly
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )

    labels = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            # Ensure word_idx does not exceed the length of ner_tags
            if word_idx < len(example["ner_tags"]):
                labels.append(example["ner_tags"][word_idx])
            else:
                labels.append(-100)
        else:
            labels.append(-100)
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=False)

# Create a 90-10 split for training and validation
tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)

print(tokenized_datasets)

label_list = [
    "O",                   # Outside of any entity
    "B-PERSON", "I-PERSON",           # Person names
    "B-LOCATION", "I-LOCATION",       # Locations
    "B-ORGANISATION", "I-ORGANISATION", # Organizations
    "B-DATE", "I-DATE",               # Dates
    "B-TIME", "I-TIME",               # Times
    "B-MONEY", "I-MONEY",             # Monetary values
    "B-PERCENTAGE", "I-PERCENTAGE",   # Percentages
    "B-FACILITY", "I-FACILITY",       # Facilities
    "B-PRODUCT", "I-PRODUCT",         # Products
    "B-EVENT", "I-EVENT",             # Events
    "B-ART", "I-ART",                 # Artworks
    "B-LAW", "I-LAW",                 # Legal documents
    "B-LANGUAGE", "I-LANGUAGE",       # Languages
    "B-GPE", "I-GPE",                 # Geopolitical entities
    "B-NORP", "I-NORP",               # Nationalities or groups
    "B-ORDINAL", "I-ORDINAL",         # Ordinal numbers
    "B-CARDINAL", "I-CARDINAL",       # Cardinal numbers
    "B-DISEASE", "I-DISEASE",         # Diseases
    "B-CONTACT", "I-CONTACT",         # Contact info
    "B-ADAGE", "I-ADAGE",             # Sayings
    "B-QUANTITY", "I-QUANTITY",       # Quantities
    "B-MISCELLANEOUS", "I-MISCELLANEOUS", # Miscellaneous entities
    "B-POSITION", "I-POSITION",       # Positions
    "B-PROJECT", "I-PROJECT"          # Projects
]

# Initialize the data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Load the model with the correct number of labels
model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=25,  # Ensure this matches the number of unique labels in `label_list`
)

# Custom metric function for evaluation
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Map back to the label list, skipping ignored tokens (-100)
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Compute metrics
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)

    # Optionally print a classification report
    print(classification_report(true_labels, true_predictions))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Define training arguments with optimized parameters
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,                    # Adjusted learning rate for stability
    per_device_train_batch_size=64,        # Adjusted for memory optimization
    per_device_eval_batch_size=64,
    num_train_epochs=8,                    # Adjusted to reduce overfitting
    weight_decay=0.01,
    fp16=True,
    gradient_accumulation_steps=4,         # Higher accumulation for smaller batch size
    logging_dir='./logs',
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"                       # Disables logging integrations (e.g., WANDB)
)

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

# Learning rate scheduler setup
num_training_steps = len(tokenized_datasets["train"]) // training_args.per_device_train_batch_size * training_args.num_train_epochs
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),  # 10% warmup steps
    num_training_steps=num_training_steps
)

# Define the Trainer with early stopping callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler),  # Pass optimizer and scheduler
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Early stopping with patience of 2
)

training_metrics = trainer.train()

eval_results = trainer.evaluate()
print(eval_results)

# Define a path to save the model
save_directory = "./XLM-RoBERTa"

# Save the model
model.save_pretrained(save_directory)

# Save the tokenizer
tokenizer.save_pretrained(save_directory)

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./XLM-RoBERTa")
model = AutoModelForTokenClassification.from_pretrained("./XLM-RoBERTa")

# Initialize the NER pipeline with GPU if available
device = 0 if torch.cuda.is_available() else -1
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)

# Define label mapping based on the label list
label_mapping = {f"LABEL_{i}": label for i, label in enumerate(label_list) if label != "O"}

# Define the function to prepare test data and evaluate
def evaluate_model(test_texts, true_labels):
    predictions = []
    for text in test_texts:
        pred_entities = nlp_ner(text)

        # Map predicted labels using the label_mapping and filter out non-entities
        pred_labels = [label_mapping.get(entity["entity_group"], "O") for entity in pred_entities if entity["entity_group"] in label_mapping]
        predictions.append(pred_labels)

    # Calculate metrics if lengths match, otherwise notify about inconsistency
    if len(true_labels) == len(predictions):
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)

        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)
        print(classification_report(true_labels, predictions))
    else:
        print("Warning: Inconsistent number of samples between true labels and predictions.")
        print(f"True Labels: {true_labels}")
        print(f"Predictions: {predictions}")

# Example test data
test_texts = ["Shahla Khuduyeva və Pasha Sığorta şirkəti haqqında məlumat."]
true_labels = [["B-PERSON", "B-ORGANISATION"]]  # True labels for the sample text
evaluate_model(test_texts, true_labels)

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./XLM-RoBERTa")
model = AutoModelForTokenClassification.from_pretrained("./XLM-RoBERTa")

# Initialize the NER pipeline with GPU if available
device = 0 if torch.cuda.is_available() else -1
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)

# Define label mapping based on the label list
label_list = [
    "O", "B-PERSON", "I-PERSON", "B-LOCATION", "I-LOCATION",
    "B-ORGANISATION", "I-ORGANISATION", "B-DATE", "I-DATE",
    "B-TIME", "I-TIME", "B-MONEY", "I-MONEY", "B-PERCENTAGE",
    "I-PERCENTAGE", "B-FACILITY", "I-FACILITY", "B-PRODUCT",
    "I-PRODUCT", "B-EVENT", "I-EVENT", "B-ART", "I-ART",
    "B-LAW", "I-LAW", "B-LANGUAGE", "I-LANGUAGE", "B-GPE",
    "I-GPE", "B-NORP", "I-NORP", "B-ORDINAL", "I-ORDINAL",
    "B-CARDINAL", "I-CARDINAL", "B-DISEASE", "I-DISEASE",
    "B-CONTACT", "I-CONTACT", "B-ADAGE", "I-ADAGE",
    "B-QUANTITY", "I-QUANTITY", "B-MISCELLANEOUS", "I-MISCELLANEOUS",
    "B-POSITION", "I-POSITION", "B-PROJECT", "I-PROJECT"
]
label_mapping = {f"LABEL_{i}": label for i, label in enumerate(label_list) if label != "O"}

# Define the function to prepare test data and evaluate
def evaluate_model(test_texts, true_labels):
    predictions = []
    for i, text in enumerate(test_texts):
        pred_entities = nlp_ner(text)

        # Map predicted labels using the label_mapping and filter out non-entities
        pred_labels = [label_mapping.get(entity["entity_group"], "O") for entity in pred_entities if entity["entity_group"] in label_mapping]

        # Adjust prediction length to match the true labels
        if len(pred_labels) != len(true_labels[i]):
            print(f"Warning: Inconsistent number of entities in sample {i+1}. Adjusting predicted entities.")
            pred_labels = pred_labels[:len(true_labels[i])]

        predictions.append(pred_labels)

    # Calculate metrics if lengths match, otherwise notify about inconsistency
    if all(len(true) == len(pred) for true, pred in zip(true_labels, predictions)):
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)

        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)
        print(classification_report(true_labels, predictions))
    else:
        print("Error: Could not align all samples correctly for evaluation.")
        print(f"True Labels: {true_labels}")
        print(f"Predictions: {predictions}")

# Example test data
test_texts = [
    "Shahla Khuduyeva və Pasha Sığorta şirkəti haqqında məlumat.",
    "İlham Əliyev Bakıda keçirilən konfransda çıxış etdi.",
    "Azərbaycan Respublikası Müdafiə Nazirliyi yeni layihə təqdim etdi.",
    "Neftçala şəhərində 10 milyondan çox pul sərf edildi.",
    "2023-cü il avqust ayında bu qərar elan edildi."
]
true_labels = [
    ["B-PERSON", "B-ORGANISATION"],
    ["B-PERSON", "B-LOCATION"],
    ["B-ORGANISATION"],
    ["B-LOCATION", "B-MONEY"],
    ["B-DATE"]
]

# Evaluate the model
evaluate_model(test_texts, true_labels)
