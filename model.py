import os
import warnings
from typing import List, Dict, Any, Optional
import numpy as np
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
    
    def preprocess_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a single example"""
        try:
            tokens = eval(example["tokens"]) if isinstance(example["tokens"], str) else example["tokens"]
            ner_tags = [int(tag) for tag in eval(example["ner_tags"])] if isinstance(example["ner_tags"], str) else example["ner_tags"]
            
            # Join tokens into text for proper tokenization
            text = " ".join(tokens)
            
            return {
                "text": text,
                "tokens": tokens,
                "ner_tags": ner_tags
            }
        except (ValueError, SyntaxError) as e:
            print(f"Error processing example {example.get('index', 'unknown')}: {e}")
            return {"text": "", "tokens": [], "ner_tags": []}
    
    def tokenize_and_align_labels(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize and align labels with tokens"""
        tokenized_inputs = self.tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_offsets_mapping=True
        )
        
        # Since we're not using is_split_into_words anymore, we need to align labels differently
        labels = []
        offset_mapping = tokenized_inputs.pop("offset_mapping")
        
        # Create token-to-label mapping
        token_start = 0
        current_token_idx = 0
        text = example["text"]
        tokens = example["tokens"]
        
        for token_idx, (start, end) in enumerate(offset_mapping):
            # Special tokens have a start index of 0 and end index of 0
            if start == end == 0:
                labels.append(-100)
                continue
                
            # Find the corresponding original token
            token_text = text[start:end]
            if token_text.strip():
                if current_token_idx < len(example["ner_tags"]):
                    labels.append(example["ner_tags"][current_token_idx])
                else:
                    labels.append(-100)
                current_token_idx += 1
            else:
                labels.append(-100)
        
        # Ensure we have the correct length
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
    
    # Preprocess dataset
    dataset = dataset.map(processor.preprocess_example)
    tokenized_datasets = dataset.map(
        processor.tokenize_and_align_labels,
        remove_columns=dataset["train"].column_names
    )
    
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
    
    # Evaluate model
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)
    
    # Save model
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