"""
Azerbaijani Named Entity Recognition (NER) Pipeline
License: CC BY-NC-ND 4.0
"""

import os
import multiprocessing
import logging
import warnings
import json
import ast
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
import sys
import traceback
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from seqeval.metrics import f1_score, precision_score, recall_score

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# Filter warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Get optimal number of workers
NUM_CPU = multiprocessing.cpu_count()
OPTIMAL_NUM_WORKERS = min(2, NUM_CPU - 1)

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

# Entity type definitions
ENTITY_TYPES = {
    0: "O",           # Outside any named entity
    1: "PERSON",      # Names of individuals
    2: "LOCATION",    # Geographical locations
    3: "ORGANISATION",# Names of companies, institutions
    4: "DATE",        # Dates or periods
    5: "TIME",        # Times of the day
    6: "MONEY",       # Monetary values
    7: "PERCENTAGE",  # Percentage values
    8: "FACILITY",    # Buildings, airports, etc.
    9: "PRODUCT",     # Products and goods
    10: "EVENT",      # Events and occurrences
    11: "ART",        # Artworks, titles of books, songs
    12: "LAW",        # Legal documents
    13: "LANGUAGE",   # Languages
    14: "GPE",        # Countries, cities, states
    15: "NORP",       # Nationalities or religious or political groups
    16: "ORDINAL",    # Ordinal numbers
    17: "CARDINAL",   # Cardinal numbers
    18: "DISEASE",    # Diseases and medical conditions
    19: "CONTACT",    # Contact information
    20: "ADAGE",      # Proverbs, sayings
    21: "QUANTITY",   # Measurements and quantities
    22: "MISCELLANEOUS", # Miscellaneous entities
    23: "POSITION",   # Professional or social positions
    24: "PROJECT"     # Names of projects or programs
}

def get_training_args(output_dir: str) -> TrainingArguments:
    """Get training arguments with optimized settings"""
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=8,  # Reduced batch size for CPU
        per_device_eval_batch_size=8,   # Reduced batch size for CPU
        num_train_epochs=5,
        weight_decay=0.01,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50,
        report_to="none",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        warmup_steps=500,
        fp16=False,  # Disable fp16 for CPU training
        dataloader_num_workers=2,
        group_by_length=True,
        gradient_accumulation_steps=4,  # Increased for CPU training
        max_grad_norm=1.0,
        # Add these for better CPU compatibility
        no_cuda=True,
        optim="adamw_torch",
    )

class AzerbaijaniNERPipeline:
    def __init__(self, model_name: str = "bert-base-multilingual-cased", output_dir: str = "az-ner-model"):
        """Initialize the NER pipeline"""
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=512,
            padding_side='right'
        )
        
        # Initialize label mappings
        self.label2id = {entity_type: idx for idx, entity_type in ENTITY_TYPES.items()}
        self.id2label = {idx: entity_type for idx, entity_type in ENTITY_TYPES.items()}
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Initialize statistics tracking
        self.stats = {
            'total_examples': 0,
            'valid_examples': 0,
            'invalid_examples': 0,
            'tag_distribution': {idx: 0 for idx in ENTITY_TYPES.keys()}
        }

    def process_row(self, row: Dict) -> Optional[Dict]:
        """Process a single data row"""
        try:
            self.stats['total_examples'] += 1
            
            # Parse tokens and tags
            tokens = row['tokens']
            tags = row['ner_tags']
            
            # Handle string representations
            if isinstance(tokens, str):
                tokens = ast.literal_eval(tokens)
            if isinstance(tags, str):
                tags = ast.literal_eval(tags)
            
            # Validate lengths match
            if len(tokens) != len(tags):
                self.stats['invalid_examples'] += 1
                return None
            
            # Validate and clean tags
            cleaned_tags = []
            for tag in tags:
                tag = int(tag) if isinstance(tag, (int, str)) and str(tag).isdigit() else 0
                if tag not in ENTITY_TYPES:
                    tag = 0
                cleaned_tags.append(tag)
                self.stats['tag_distribution'][tag] += 1
            
            self.stats['valid_examples'] += 1
            return {
                'tokens': tokens,
                'ner_tags': cleaned_tags
            }
        except Exception as e:
            self.stats['invalid_examples'] += 1
            return None

    def load_dataset(self, data_path: str) -> DatasetDict:
        """Load and prepare the dataset"""
        logging.info(f"Loading dataset from {data_path}")
        
        # Reset statistics
        self.stats = {
            'total_examples': 0,
            'valid_examples': 0,
            'invalid_examples': 0,
            'tag_distribution': {idx: 0 for idx in ENTITY_TYPES.keys()}
        }
        
        try:
            # Load data
            df = pd.read_parquet(data_path)
            logging.info(f"Loaded {len(df)} rows from {data_path}")
            
            # Process rows
            processed_data = []
            for _, row in df.iterrows():
                processed_row = self.process_row(row)
                if processed_row is not None:
                    processed_data.append(processed_row)
            
            # Create dataset
            dataset = Dataset.from_pandas(
                pd.DataFrame(processed_data),
                features=Features({
                    'tokens': Sequence(Value('string')),
                    'ner_tags': Sequence(Value('int64'))
                })
            )
            
            # Create splits
            train_test = dataset.train_test_split(test_size=0.2, seed=42)
            test_valid = train_test['test'].train_test_split(test_size=0.5, seed=42)
            
            dataset_dict = DatasetDict({
                'train': train_test['train'],
                'validation': test_valid['train'],
                'test': test_valid['test']
            })
            
            # Log statistics
            self._log_statistics(dataset_dict)
            
            return dataset_dict
            
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            raise

    def _log_statistics(self, dataset_dict: DatasetDict):
        """Log dataset statistics"""
        logging.info("\nDataset Statistics:")
        logging.info(f"Total examples processed: {self.stats['total_examples']}")
        logging.info(f"Valid examples: {self.stats['valid_examples']}")
        logging.info(f"Invalid examples: {self.stats['invalid_examples']}")
        
        logging.info("\nTag Distribution:")
        for tag_id, count in self.stats['tag_distribution'].items():
            logging.info(f"{ENTITY_TYPES[tag_id]}: {count}")
        
        logging.info("\nDataset Splits:")
        for split, ds in dataset_dict.items():
            logging.info(f"{split} set size: {len(ds)}")

    def tokenize_and_align_labels(self, examples: Dict) -> Dict:
        """Tokenize and align labels with tokens"""
        max_length = 256 if self.device.type == "cpu" else 512
        
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
            padding="max_length"
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def compute_metrics(self, eval_preds: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Compute evaluation metrics with detailed entity analysis"""
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=2)

        # Convert indices to tag names
        true_predictions = [
            [ENTITY_TYPES[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [ENTITY_TYPES[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Overall metrics
        metrics = {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions)
        }

        # Per-entity type metrics
        for entity_type in set(ENTITY_TYPES.values()) - {'O'}:
            entity_preds = [[p == entity_type for p in pred] for pred in true_predictions]
            entity_labels = [[l == entity_type for l in label] for label in true_labels]
            
            try:
                metrics[f"{entity_type}_f1"] = f1_score(entity_labels, entity_preds)
            except:
                metrics[f"{entity_type}_f1"] = 0.0

        return metrics
    

    def train(self, dataset_dict: DatasetDict) -> Trainer:
        """Train the model with checkpointing and validation"""
        logging.info("Initializing model...")
        
        if self.device.type == "cpu":
            logging.warning("Training on CPU. This might be slow!")
            
        # Clear GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize model
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(ENTITY_TYPES),
            id2label=self.id2label,
            label2id=self.label2id,
            torch_dtype=torch.float32
        ).to(self.device)

        # Log model architecture
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

        # Prepare datasets
        tokenized_datasets = dataset_dict.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset_dict["train"].column_names,
            num_proc=OPTIMAL_NUM_WORKERS,
            load_from_cache_file=False
        )

        # Get training arguments
        training_args = get_training_args(str(self.output_dir))
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForTokenClassification(
                self.tokenizer,
                pad_to_multiple_of=8
            ),
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.01
                )
            ]
        )

        # Train
        try:
            train_result = trainer.train()
            metrics = train_result.metrics
            
            # Save training metrics
            trainer.save_metrics("train", metrics)
            trainer.save_model(str(self.output_dir))
            
            # Save tokenizer and config
            self.tokenizer.save_pretrained(str(self.output_dir))
            
            # Log final metrics
            logging.info(f"Training metrics: {metrics}")
            
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            raise

        return trainer
        
def main():
    """Main function to run the pipeline"""
    start_time = datetime.now()
    
    try:
        # Initialize pipeline
        pipeline = AzerbaijaniNERPipeline()
        
        # Set data path
        data_path = "train-00000-of-00001.parquet"
        
        # Load and process dataset
        dataset_dict = pipeline.load_dataset(data_path)
        
        # Train model
        trainer = pipeline.train(dataset_dict)
        
        # Evaluate
        test_results = trainer.evaluate(
            dataset_dict["test"].map(
                pipeline.tokenize_and_align_labels,
                batched=True,
                remove_columns=dataset_dict["test"].column_names
            )
        )
        
        # Add timestamp to results
        test_results["timestamp"] = datetime.now().isoformat()
        test_results["training_duration"] = str(datetime.now() - start_time)
        
        # Save results with timestamp
        results_path = pipeline.output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Training completed successfully.")
        logging.info(f"Results saved to {results_path}")
        logging.info(f"Total training time: {datetime.now() - start_time}")
        
    except Exception as e:
        logging.error(f"Pipeline error: {str(e)}", exc_info=True)
        raise
        
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
if __name__ == "__main__":
    main()