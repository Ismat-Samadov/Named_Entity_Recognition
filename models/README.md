# Azerbaijani Named Entity Recognition (NER) with XLM-RoBERTa

This project fine-tunes a custom NER model for Azerbaijani text using the multilingual XLM-RoBERTa model. This notebook and its supporting files enable extracting named entities like **persons**, **locations**, **organizations**, and **dates** from Azerbaijani text.

### Notebook Source
This notebook was created in Google Colab and can be accessed [here](https://colab.research.google.com/drive/1EYYZa7dya2RjTZXHSJ4pzIOgzqR8lmSk).

## Setup Instructions

1. **Install Required Libraries**:
   The following packages are necessary for running this notebook:
   ```bash
   pip install transformers datasets seqeval huggingface_hub
   ```

2. **Hugging Face Hub Authentication**:
   Set up Hugging Face Hub authentication to save and manage your trained models:
   ```python
   from huggingface_hub import login
   login(token="YOUR_HUGGINGFACE_TOKEN")
   ```
   Replace `YOUR_HUGGINGFACE_TOKEN` with your Hugging Face token.

3. **Disable Unnecessary Warnings**:
   For a cleaner output, some warnings are disabled:
   ```python
   import os
   import warnings

   os.environ["WANDB_DISABLED"] = "true"
   warnings.filterwarnings("ignore")
   ```

## Detailed Code Walkthrough

### 1. **Data Loading and Preprocessing**

#### Loading the Azerbaijani NER Dataset
The dataset for Azerbaijani NER is loaded from the Hugging Face Hub:
```python
from datasets import load_dataset

dataset = load_dataset("LocalDoc/azerbaijani-ner-dataset")
print(dataset)
```
This dataset contains Azerbaijani texts labeled with NER tags.

#### Preprocessing Tokens and NER Tags
To ensure compatibility, the tokens and NER tags are processed using the `ast` module:
```python
import ast

def preprocess_example(example):
    try:
        example["tokens"] = ast.literal_eval(example["tokens"])
        example["ner_tags"] = list(map(int, ast.literal_eval(example["ner_tags"])))
    except (ValueError, SyntaxError) as e:
        print(f"Skipping malformed example: {example['index']} due to error: {e}")
        example["tokens"] = []
        example["ner_tags"] = []
    return example

dataset = dataset.map(preprocess_example)
```
This function checks each example for format correctness, converting strings to lists of tokens and tags.

### 2. **Tokenization and Label Alignment**

#### Initializing the Tokenizer
The `AutoTokenizer` class is used to initialize the XLM-RoBERTa tokenizer:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
```

#### Tokenization and Label Alignment
Each token is aligned with its label using a custom function:
```python
def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(
        example["tokens"],
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
            labels.append(example["ner_tags"][word_idx] if word_idx < len(example["ner_tags"]) else -100)
        else:
            labels.append(-100)
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=False)
```
Tokens and labels are aligned, with `-100` used to ignore sub-tokens created during tokenization.

### 3. **Dataset Split for Training and Validation**
The dataset is split into training and validation sets:
```python
tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)
```
This ensures a 90-10 split, maintaining a consistent setup for training and testing.

### 4. **Define Labels and Model Components**

#### Define Label List
The NER tags are set up as BIO-tagging (Begin, Inside, Outside):
```python
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
```

#### Initialize Model and Data Collator
The model and data collator are set up for token classification:
```python
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=len(label_list)
)

data_collator = DataCollatorForTokenClassification(tokenizer)
```

### 5. **Define Evaluation Metrics**

The model’s performance is evaluated based on precision, recall, and F1 score:
```python
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }
```

### 6. **Training Setup and Execution**

#### Set Training Parameters
The `TrainingArguments` define configurations for model training:
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=8,
    weight_decay=0.01,
    fp16=True,
    logging_dir='./logs',
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"
)
```

#### Initialize Trainer and Train the Model
The `Trainer` class handles training and evaluation:
```python
from transformers import Trainer, EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

training_metrics = trainer.train()
eval_results = trainer.evaluate()
print(eval_results)
```

### 7. **Save the Trained Model**

After training, save the model and tokenizer for later use:
```python
save_directory = "./XLM-RoBERTa"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
```

### 8. **Inference with the NER Pipeline**

#### Initialize the NER Pipeline
The pipeline provides a high-level API for NER:
```python
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)
```

#### Custom Evaluation Function
The `evaluate_model` function allows testing on custom sentences:
```python
label_mapping = {f"LABEL_{i}": label for i, label in enumerate(label_list) if label != "O"}

def evaluate_model(test_texts, true_labels):
    predictions = []
    for i, text in enumerate(test_texts):
        pred_entities = nlp_ner(text)
        pred_labels = [label_mapping.get(entity["entity_group"], "O

") for entity in pred_entities if entity["entity_group"] in label_mapping]
        if len(pred_labels) != len(true_labels[i]):
            print(f"Warning: Inconsistent number of entities in sample {i+1}. Adjusting predicted entities.")
            pred_labels = pred_labels[:len(true_labels[i])]
        predictions.append(pred_labels)
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
```

#### Test on a Sample Sentence
An example test with expected output labels:
```python
test_texts = ["Shahla Khuduyeva və Pasha Sığorta şirkəti haqqında məlumat."]
true_labels = [["B-PERSON", "B-ORGANISATION"]]
evaluate_model(test_texts, true_labels)
```