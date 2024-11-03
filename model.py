import os
import warnings
from typing import List, Dict, Any, Optional
import numpy as np
import ast
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModel,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torchcrf import CRF
from seqeval.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    accuracy_score as seqeval_accuracy
)
from sklearn.metrics import accuracy_score
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for BiLSTM-CRF model"""
    hidden_dim: int = 256
    dropout: float = 0.1
    num_lstm_layers: int = 2
    base_model: str = "xlm-roberta-base"
    max_length: int = 128

class BiLSTMCRFNER(nn.Module):
    """BiLSTM-CRF model for Named Entity Recognition"""
    
    def __init__(self, num_labels: int, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Transformer encoder
        self.transformer = AutoModel.from_pretrained(config.base_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=self.transformer.config.hidden_size,
            hidden_size=config.hidden_dim,
            num_layers=config.num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0
        )
        
        # Linear layer
        self.hidden2label = nn.Linear(config.hidden_dim * 2, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get transformer embeddings
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        
        # BiLSTM processing
        lstm_out, _ = self.bilstm(sequence_output)
        lstm_out = self.dropout(lstm_out)
        
        # Get emissions for CRF
        emissions = self.hidden2label(lstm_out)
        
        if labels is not None:
            # Training mode
            log_likelihood = self.crf(emissions, labels, mask=attention_mask.byte())
            return -log_likelihood
        else:
            # Inference mode
            return self.crf.decode(emissions, mask=attention_mask.byte())

class NERDataProcessor:
    """Handles data preprocessing for NER tasks"""
    
    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def safe_eval(self, string_data: str) -> List[str]:
        """Safely evaluate string representation of list"""
        try:
            if isinstance(string_data, str):
                # Remove any unwanted whitespace and ensure proper quotes
                cleaned_str = string_data.strip().replace("'", '"')
                return ast.literal_eval(cleaned_str)
            return string_data
        except (ValueError, SyntaxError) as e:
            print(f"Error evaluating string: {e}")
            return []

    def preprocess_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a single example"""
        try:
            # Safely convert tokens and tags
            tokens = self.safe_eval(example["tokens"])
            ner_tags = self.safe_eval(example["ner_tags"])
            
            # Convert tags to integers if they're strings
            if ner_tags and isinstance(ner_tags[0], str):
                ner_tags = [int(tag) for tag in ner_tags]
            
            # Ensure tokens is a list of strings
            if not tokens or not isinstance(tokens, list):
                print(f"Invalid tokens format: {tokens}")
                return {"text": "", "tokens": [], "ner_tags": []}
            
            # Join tokens into text
            text = " ".join(str(token) for token in tokens)
            
            return {
                "text": text,
                "tokens": tokens,
                "ner_tags": ner_tags
            }
            
        except Exception as e:
            print(f"Error processing example {example.get('index', 'unknown')}: {e}")
            print(f"Tokens: {example.get('tokens')}")
            print(f"NER tags: {example.get('ner_tags')}")
            return {"text": "", "tokens": [], "ner_tags": []}
    
    def tokenize_and_align_labels(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize and align labels with tokens"""
        # Skip empty examples
        if not example["text"]:
            return {
                "input_ids": [0] * self.max_length,
                "attention_mask": [0] * self.max_length,
                "labels": [-100] * self.max_length
            }
        
        tokenized_inputs = self.tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_offsets_mapping=True
        )
        
        labels = []
        offset_mapping = tokenized_inputs.pop("offset_mapping")
        
        current_token_idx = 0
        current_token = example["tokens"][0] if example["tokens"] else ""
        current_token_start = 0
        
        for start, end in offset_mapping:
            # Special tokens
            if start == end == 0:
                labels.append(-100)
                continue
            
            # Regular tokens
            token_text = example["text"][start:end]
            
            if current_token_idx < len(example["ner_tags"]):
                if token_text.strip():
                    labels.append(example["ner_tags"][current_token_idx])
                    if end - start >= len(current_token):
                        current_token_idx += 1
                        if current_token_idx < len(example["tokens"]):
                            current_token = example["tokens"][current_token_idx]
                else:
                    labels.append(-100)
            else:
                labels.append(-100)
        
        # Ensure correct length
        if len(labels) < self.max_length:
            labels.extend([-100] * (self.max_length - len(labels)))
        elif len(labels) > self.max_length:
            labels = labels[:self.max_length]
            
        tokenized_inputs["labels"] = labels
        return tokenized_inputs


def create_training_args(
    output_dir: str,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    num_epochs: int = 5
) -> TrainingArguments:
    """Create training arguments for the Trainer"""
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        fp16=True,
        optim="adamw_torch",
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        dataloader_num_workers=4,
        lr_scheduler_type="cosine_with_restarts",
        warmup_ratio=0.1
    )

def compute_metrics(pred_obj: Any) -> Dict[str, float]:
    """Compute evaluation metrics"""
    predictions, labels = pred_obj
    predictions = [p for batch in predictions for p in batch]
    labels = [l for batch in labels for l in batch]
    
    # Convert to entity labels
    true_labels = [[LABEL_LIST[l] for l in label if l != -100] for label in labels]
    true_predictions = [[LABEL_LIST[p] for p in pred] for pred in predictions]
    
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "token_accuracy": accuracy_score(
            [l for sublist in true_labels for l in sublist],
            [p for sublist in true_predictions for p in sublist]
        ),
        "entity_accuracy": seqeval_accuracy(true_labels, true_predictions),
        "macro_f1": f1_score(true_labels, true_predictions, average="macro"),
        "micro_f1": f1_score(true_labels, true_predictions, average="micro")
    }

def train_model(
    model: BiLSTMCRFNER,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: AutoTokenizer,
    training_args: TrainingArguments
) -> Trainer:
    """Train the model"""
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Validate batch shapes before training
    for batch in trainer.get_train_dataloader():
        assert batch["input_ids"].shape == batch["labels"].shape, \
            f"Shape mismatch: {batch['input_ids'].shape} vs {batch['labels'].shape}"
    
    trainer.train()
    return trainer



def main():
    # Initialize configurations
    model_config = ModelConfig()
    
    # Load dataset
    dataset = load_dataset("LocalDoc/azerbaijani-ner-dataset")
    
    # Initialize tokenizer and data processor
    tokenizer = AutoTokenizer.from_pretrained(model_config.base_model)
    processor = NERDataProcessor(tokenizer, model_config.max_length)
    
    # Print example data for debugging
    print("Sample raw example:")
    print(dataset["train"][0])
    
    # Preprocess dataset
    preprocessed_dataset = dataset.map(
        processor.preprocess_example,
        remove_columns=dataset["train"].column_names
    )
    
    # Print preprocessed example for debugging
    print("\nSample preprocessed example:")
    print(preprocessed_dataset["train"][0])
    
    # Tokenize and align labels
    tokenized_datasets = preprocessed_dataset.map(
        processor.tokenize_and_align_labels,
        remove_columns=preprocessed_dataset["train"].column_names
    )
    
    # Print tokenized example for debugging
    print("\nSample tokenized example:")
    print(tokenized_datasets["train"][0])
    
    # Split dataset
    train_test_split = tokenized_datasets["train"].train_test_split(test_size=0.1)
    
    # Initialize model
    model = BiLSTMCRFNER(
        num_labels=len(LABEL_LIST),
        config=model_config
    )
    
    # Train model
    training_args = create_training_args("./results")
    trainer = train_model(
        model,
        train_test_split["train"],
        train_test_split["test"],
        tokenizer,
        training_args
    )
    
    # Evaluate and save model
    eval_results = trainer.evaluate()
    print("\nEvaluation Results:", eval_results)
    
    save_directory = "./XLM-RoBERTa-BiLSTM-CRF"
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
if __name__ == "__main__":
    # Set environment variables and warnings
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    warnings.filterwarnings("ignore")
    
    # Define labels
    LABEL_LIST = [
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
    
    main()