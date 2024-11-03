# Constants for label mapping
LABEL_MAP = {
    "O": 0,
    "PERSON": 1,
    "LOCATION": 2,
    "ORGANISATION": 3,
    "DATE": 4,
    "TIME": 5,
    "MONEY": 6,
    "PERCENTAGE": 7,
    "FACILITY": 8,
    "PRODUCT": 9,
    "EVENT": 10,
    "ART": 11,
    "LAW": 12,
    "LANGUAGE": 13,
    "GPE": 14,
    "NORP": 15,
    "ORDINAL": 16,
    "CARDINAL": 17,
    "DISEASE": 18,
    "CONTACT": 19,
    "ADAGE": 20,
    "QUANTITY": 21,
    "MISCELLANEOUS": 22,
    "POSITION": 23,
    "PROJECT": 24
}

# Inverse mapping for predictions
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

class NERDataProcessor:
    """Handles data preprocessing for NER tasks"""
    
    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def safe_eval(self, string_data: str) -> List[str]:
        """Safely evaluate string representation of list"""
        try:
            if isinstance(string_data, str):
                # Clean the string - handle potential JSON-like formatting
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
            
            # Convert tags to integers
            if ner_tags and isinstance(ner_tags[0], str):
                # Handle potential BIO format
                processed_tags = []
                for tag in ner_tags:
                    if tag == "O":
                        processed_tags.append(0)
                    else:
                        # Handle BIO format (B-PERSON, I-PERSON, etc.)
                        parts = tag.split("-")
                        if len(parts) == 2:
                            bio_prefix, entity_type = parts
                            if entity_type in LABEL_MAP:
                                processed_tags.append(LABEL_MAP[entity_type])
                            else:
                                processed_tags.append(0)  # Default to O if unknown
                        else:
                            processed_tags.append(0)  # Default to O if malformed
                ner_tags = processed_tags
            
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

def compute_metrics(pred_obj: Any) -> Dict[str, float]:
    """Compute evaluation metrics"""
    predictions, labels = pred_obj
    predictions = [p for batch in predictions for p in batch]
    labels = [l for batch in labels for l in batch]
    
    # Convert to entity labels
    true_labels = [[ID2LABEL[l] for l in label if l != -100] for label in labels]
    true_predictions = [[ID2LABEL[p] for p in pred] for pred in predictions]
    
    # Calculate metrics
    results = {
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
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, true_predictions))
    
    return results

def main():
    # Initialize configurations
    model_config = ModelConfig()
    
    # Load dataset
    dataset = load_dataset("LocalDoc/azerbaijani-ner-dataset")
    
    # Print dataset info
    print("\nDataset Info:")
    print(f"Number of examples: {len(dataset['train'])}")
    print("\nSample raw example:")
    print(dataset["train"][0])
    
    # Initialize tokenizer and data processor
    tokenizer = AutoTokenizer.from_pretrained(model_config.base_model)
    processor = NERDataProcessor(tokenizer, model_config.max_length)
    
    # Preprocess dataset
    preprocessed_dataset = dataset.map(
        processor.preprocess_example,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing dataset"
    )
    
    # Print preprocessed example
    print("\nSample preprocessed example:")
    print(preprocessed_dataset["train"][0])
    
    # Tokenize and align labels
    tokenized_datasets = preprocessed_dataset.map(
        processor.tokenize_and_align_labels,
        remove_columns=preprocessed_dataset["train"].column_names,
        desc="Tokenizing dataset"
    )
    
    # Initialize model
    model = BiLSTMCRFNER(
        num_labels=len(LABEL_MAP),
        config=model_config
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create training arguments
    training_args = create_training_args(
        output_dir="./results",
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=5
    )
    
    # Split dataset
    train_test_split = tokenized_datasets["train"].train_test_split(
        test_size=0.1,
        seed=42
    )
    
    # Train model
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
    
    # Save model and tokenizer
    save_directory = "./XLM-RoBERTa-BiLSTM-CRF"
    save_model(model, tokenizer, save_directory)

if __name__ == "__main__":
    main()