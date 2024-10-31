import os
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import f1_score, precision_score, recall_score
import torch
import json
import ast
from typing import List, Dict, Tuple

class AzerbaijaniNERPipeline:
    def __init__(self, model_name="bert-base-multilingual-cased", output_dir="az-ner-model"):
        self.model_name = model_name
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.initialize_label_mappings()
        
    def initialize_label_mappings(self):
        """Initialize label mappings for the NER tags"""
        self.label2id = {str(i): i for i in range(25)}  # 0-24 for all entity types
        self.id2label = {v: k for k, v in self.label2id.items()}

    def parse_list_string(self, s: str) -> List:
        """Parse a string representation of a list"""
        try:
            if pd.isna(s) or not isinstance(s, str):
                return []
            result = ast.literal_eval(s)
            if not isinstance(result, list):
                return []
            return result
        except:
            return []

    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset"""
        print("Cleaning and validating data...")
        
        def process_row(row):
            try:
                # Parse tokens and tags
                tokens = self.parse_list_string(row['tokens'])
                ner_tags = self.parse_list_string(row['ner_tags'])
                
                # Skip invalid rows
                if not tokens or not ner_tags or len(tokens) != len(ner_tags):
                    return None
                
                # Ensure all tags are integers and within valid range
                ner_tags = [
                    int(tag) if isinstance(tag, (int, str)) and str(tag).isdigit() and int(tag) < 25 
                    else 0 
                    for tag in ner_tags
                ]
                
                return {
                    'tokens': tokens,
                    'ner_tags': ner_tags,
                }
            except Exception as e:
                return None

        # Process all rows
        processed_data = []
        skipped_rows = 0
        
        for _, row in df.iterrows():
            processed_row = process_row(row)
            if processed_row is not None:
                processed_data.append(processed_row)
            else:
                skipped_rows += 1
        
        print(f"Skipped {skipped_rows} invalid rows")
        print(f"Processed {len(processed_data)} valid rows")
        
        return pd.DataFrame(processed_data)

    def create_features(self) -> Features:
        """Create feature descriptions for the dataset"""
        return Features({
            'tokens': Sequence(Value('string')),
            'ner_tags': Sequence(Value('int64'))
        })

    def load_dataset(self, parquet_path: str) -> DatasetDict:
        """Load and prepare the dataset"""
        print(f"Loading dataset from {parquet_path}...")
        
        # Load parquet file
        df = pd.read_parquet(parquet_path)
        print(f"Initial dataset size: {len(df)} rows")
        
        # Clean and validate data
        processed_df = self.clean_and_validate_data(df)
        
        # Create dataset with explicit feature definitions
        dataset = Dataset.from_pandas(
            processed_df,
            features=self.create_features(),
            preserve_index=False
        )
        
        # Split dataset
        train_test = dataset.train_test_split(test_size=0.2, seed=42)
        test_valid = train_test['test'].train_test_split(test_size=0.5, seed=42)
        
        dataset_dict = DatasetDict({
            'train': train_test['train'],
            'validation': test_valid['train'],
            'test': test_valid['test']
        })
        
        # Print split sizes and sample
        print("\nDataset splits:")
        for split, ds in dataset_dict.items():
            print(f"{split} set size: {len(ds)} examples")
        
        print("\nSample from training set:")
        sample = dataset_dict['train'][0]
        print(f"Tokens: {sample['tokens']}")
        print(f"Tags: {sample['ner_tags']}")
        
        # Calculate and print label distribution
        print("\nLabel distribution in training set:")
        all_labels = []
        for example in dataset_dict['train']:
            all_labels.extend(example['ner_tags'])
        label_counts = pd.Series(all_labels).value_counts().sort_index()
        for label, count in label_counts.items():
            print(f"Label {label}: {count} occurrences")
        
        return dataset_dict

    def tokenize_and_align_labels(self, examples):
        """Tokenize and align labels with tokens"""
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=512,
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
                    label_ids.append(int(label[word_idx]))
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
                
            labels.append(label_ids)
            
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, eval_preds):
        """Compute evaluation metrics"""
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (-100)
        true_predictions = [
            [str(p) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [str(l) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions)
        }

    def train(self, dataset_dict: DatasetDict):
        """Train the NER model"""
        print("Initializing model...")
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        print("Preparing datasets...")
        tokenized_datasets = dataset_dict.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset_dict["train"].column_names
        )
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="steps",
            eval_steps=100,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            push_to_hub=False,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=os.path.join(self.output_dir, 'logs'),
            logging_steps=50,
            report_to="none"  # Disable wandb logging
        )
        
        print("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
            compute_metrics=self.compute_metrics
        )
        
        print("Starting training...")
        trainer.train()
        
        print("Saving model...")
        trainer.save_model(self.output_dir)
        
        return trainer

def main():
    # Initialize pipeline
    pipeline = AzerbaijaniNERPipeline()
    
    # Load and process dataset
    dataset_dict = pipeline.load_dataset("train-00000-of-00001.parquet")
    
    # Train model
    trainer = pipeline.train(dataset_dict)
    
    # Final evaluation
    print("Performing final evaluation...")
    test_results = trainer.evaluate(
        dataset_dict["test"].map(
            pipeline.tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset_dict["test"].column_names
        )
    )
    print("\nFinal Test Results:", json.dumps(test_results, indent=2))

if __name__ == "__main__":
    main()