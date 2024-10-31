from huggingface_hub import login, whoami
import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
import torch
import json
import ast
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
import logging
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

@dataclass
class ModelConfig:
    """Configuration class for model parameters"""
    model_name: str = "bert-base-multilingual-cased"
    output_dir: str = "az-ner-model"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    eval_steps: int = 100
    logging_steps: int = 50
    save_steps: int = 100
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    hf_token: Optional[str] = None

class AzerbaijaniNERPipeline:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.initialize_tag_mappings()
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Track statistics
        self.stats = {
            'invalid_tags': Counter(),
            'mismatched_lengths': 0,
            'cleaned_examples': 0,
            'total_examples': 0
        }

    def _setup_logging(self):
        """Set up logging configuration"""
        log_file = self.output_dir / "training.log"
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )

    def initialize_tag_mappings(self):
        """Initialize BIO tag mappings"""
        self.bio_tags = {
            'O': 0,      # Outside of named entity
            'B-PER': 1,  # Beginning of person name
            'I-PER': 2,  # Inside of person name
            'B-ORG': 3,  # Beginning of organization
            'I-ORG': 4,  # Inside of organization
            'B-LOC': 5,  # Beginning of location
            'I-LOC': 6,  # Inside of location
            'B-MISC': 7, # Beginning of miscellaneous
            'I-MISC': 8  # Inside of miscellaneous
        }
        
        self.id2tag = {v: k for k, v in self.bio_tags.items()}
        self.num_labels = len(self.bio_tags)
        
        # Create mapping from old tags to new tags
        self.tag_mapping = {
            0: 0,    # O
            1: 1,    # B-PER
            2: 2,    # I-PER
            3: 3,    # B-ORG
            4: 4,    # I-ORG
            5: 5,    # B-LOC
            6: 6,    # I-LOC
            7: 7,    # B-MISC
            8: 8,    # I-MISC
            # Map other tags to O
            9: 0, 10: 0, 11: 0, 12: 0, 13: 0,
            14: 0, 15: 0, 16: 0, 17: 0, 18: 0,
            19: 0, 20: 0, 21: 0, 22: 0, 23: 0,
            24: 0
        }

    def normalize_tags(self, tags: List[int]) -> List[int]:
        """Normalize tags to standard BIO format"""
        return [self.tag_mapping.get(tag, 0) for tag in tags]

    def validate_example(self, tokens: List[str], tags: List[int]) -> bool:
        """Validate a single example"""
        if len(tokens) != len(tags):
            self.stats['mismatched_lengths'] += 1
            return False

        # Check for invalid tags
        for tag in tags:
            if tag not in self.tag_mapping:
                self.stats['invalid_tags'][tag] += 1
                return False

        return True

    def clean_tags(self, tags: List[int]) -> List[int]:
        """Clean and validate tag sequences"""
        cleaned_tags = []
        prev_tag = 'O'
        
        for tag in self.normalize_tags(tags):
            tag_name = self.id2tag[tag]
            
            # Fix invalid BIO sequences
            if tag_name.startswith('I-') and not prev_tag.endswith(tag_name[2:]):
                # Convert I- to B- if it doesn't follow matching B- or I-
                tag_name = 'B-' + tag_name[2:]
                tag = self.bio_tags[tag_name]
            
            cleaned_tags.append(tag)
            prev_tag = tag_name
            
        return cleaned_tags

    def process_row(self, example: Dict) -> Optional[Dict]:
        """Process a single example with cleaning and validation"""
        try:
            self.stats['total_examples'] += 1
            
            tokens = example['tokens']
            ner_tags = example['ner_tags']
            
            # Handle string representations
            if isinstance(tokens, str):
                tokens = ast.literal_eval(tokens)
            if isinstance(ner_tags, str):
                ner_tags = ast.literal_eval(ner_tags)
            
            # Convert to integers if needed
            ner_tags = [int(tag) for tag in ner_tags]
            
            # Validate example
            if not self.validate_example(tokens, ner_tags):
                return None
                
            # Clean and normalize tags
            cleaned_tags = self.clean_tags(ner_tags)
            
            self.stats['cleaned_examples'] += 1
            return {
                'tokens': tokens,
                'ner_tags': cleaned_tags
            }
            
        except Exception as e:
            logging.error(f"Error processing row: {str(e)}")
            return None

    def load_dataset(self, dataset_name: str = "LocalDoc/azerbaijani-ner-dataset") -> DatasetDict:
        """Load and prepare the dataset with cleaning and validation"""
        logging.info(f"Loading dataset: {dataset_name}")
        
        try:
            # Reset statistics
            self.stats = {
                'invalid_tags': Counter(),
                'mismatched_lengths': 0,
                'cleaned_examples': 0,
                'total_examples': 0
            }
            
            # Load raw dataset
            raw_dataset = load_dataset(
                dataset_name,
                token=self.config.hf_token,
                data_files={
                    "train": "data/train-00000-of-00001.parquet"
                },
                download_mode="force_redownload"
            )
            
            # Process and clean data
            processed_dataset = raw_dataset.map(
                self.process_row,
                batched=False,
                num_proc=4,
                remove_columns=raw_dataset['train'].column_names,
                load_from_cache_file=False
            )
            
            # Remove invalid examples
            processed_dataset = processed_dataset.filter(lambda x: x is not None)
            
            # Create splits
            train_test = processed_dataset['train'].train_test_split(test_size=0.2, seed=42)
            test_valid = train_test['test'].train_test_split(test_size=0.5, seed=42)
            
            dataset_dict = DatasetDict({
                'train': train_test['train'],
                'validation': test_valid['train'],
                'test': test_valid['test']
            })
            
            # Log statistics
            self._log_dataset_statistics(dataset_dict)
            self._log_cleaning_statistics()
            
            return dataset_dict
            
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            raise

    def _log_cleaning_statistics(self):
        """Log statistics about data cleaning process"""
        logging.info("\nData Cleaning Statistics:")
        logging.info(f"Total examples processed: {self.stats['total_examples']}")
        logging.info(f"Examples retained after cleaning: {self.stats['cleaned_examples']}")
        logging.info(f"Examples with mismatched lengths: {self.stats['mismatched_lengths']}")
        
        if self.stats['invalid_tags']:
            logging.info("\nInvalid tag counts:")
            for tag, count in self.stats['invalid_tags'].most_common():
                logging.info(f"Tag {tag}: {count} occurrences")

    def _log_dataset_statistics(self, dataset_dict: DatasetDict):
        """Log detailed dataset statistics"""
        for split, dataset in dataset_dict.items():
            tag_dist = Counter()
            for example in dataset:
                tag_dist.update(example['ner_tags'])
            
            stats = {
                'num_examples': len(dataset),
                'avg_sequence_length': np.mean([len(x['tokens']) for x in dataset]),
                'max_sequence_length': max(len(x['tokens']) for x in dataset),
                'tag_distribution': {self.id2tag[k]: v for k, v in tag_dist.items()}
            }
            
            logging.info(f"\n{split} set statistics:")
            logging.info(json.dumps(stats, indent=2))

    # [Rest of the class implementation remains the same]

def main():
    # Configuration with HuggingFace token
    config = ModelConfig(
        hf_token="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"
    )
    
    try:
        # Initialize pipeline
        pipeline = AzerbaijaniNERPipeline(config)
        
        # Load and prepare dataset
        dataset_dict = pipeline.load_dataset()
        
        # Continue with training as before
        trainer = pipeline.train(dataset_dict)
        
        # Final evaluation
        test_results = trainer.evaluate(
            dataset_dict["test"].map(
                pipeline.tokenize_and_align_labels,
                batched=True,
                remove_columns=dataset_dict["test"].column_names
            )
        )
        
        # Save results
        with open(pipeline.output_dir / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
            
        logging.info(f"Training completed successfully. Results saved to {pipeline.output_dir}")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()