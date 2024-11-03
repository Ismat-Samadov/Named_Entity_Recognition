import ast

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