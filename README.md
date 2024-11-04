# Named_Entity_Recognition

### Custom Named Entity Recognition (NER) Model for Azerbaijani Language

This project provides a custom Named Entity Recognition (NER) model tailored for the Azerbaijani language. It includes a FastAPI application for deploying the model, as well as a frontend interface to test and view the NER results.

### Demo

You can try out the deployed model here: [Named Entity Recognition Demo](https://named-entity-recognition.fly.dev/)

**Note:** The server is hosted on a free tier, so it may take 1-2 minutes to wake up if it’s inactive when you access it. Please be patient as the server starts up.

## File Structure

```plaintext
.
├── Dockerfile                # Defines instructions for building the Docker image
├── README.md                 # Project overview, setup, and usage instructions
├── fly.toml                  # Configuration file for Fly.io deployment
├── main.py                   # Main FastAPI app file handling API endpoints and model loading
├── models                    # Contains model-related notebooks, scripts, and data
│   ├── XLM-RoBERTa.ipynb     # Notebook for XLM-RoBERTa model training/testing
│   ├── mBERT.ipynb           # Notebook for mBERT model training/testing
│   ├── push_to_HF.py         # Script to push model to Hugging Face hub
│   └── train-00000-of-00001.parquet  # Parquet file with model training/evaluation data
├── requirements.txt          # Lists all Python dependencies for the project
├── static                    # Contains frontend assets (JavaScript, CSS)
│   ├── app.js                # JavaScript for handling frontend functionality
│   └── style.css             # CSS for styling the frontend interface
└── templates                 # HTML templates for rendering the frontend interface
    └── index.html            # Main HTML file for the user interface
```

## Data and Model Links

- **Dataset**: [Azerbaijani NER Dataset](https://huggingface.co/datasets/LocalDoc/azerbaijani-ner-dataset)
- **mBERT Model**: [mBERT Azerbaijani NER](https://huggingface.co/IsmatS/mbert-az-ner)
- **XLM-RoBERTa Model**: [XLM-RoBERTa Azerbaijani NER](https://huggingface.co/IsmatS/xlm-roberta-az-ner)

Both models were fine-tuned on a premium A100 GPU in Google Colab for optimized training performance.

**Note**: Due to its superior performance, the XLM-RoBERTa model was selected for deployment.

## Model Performance Metrics

### mBERT Model

| Epoch | Training Loss | Validation Loss | Precision | Recall   | F1       | Accuracy |
|-------|---------------|----------------|-----------|----------|----------|----------|
| 1     | 0.295200      | 0.265711       | 0.715424  | 0.622853 | 0.665937 | 0.919136 |
| 2     | 0.248600      | 0.252083       | 0.721036  | 0.637979 | 0.676970 | 0.921439 |
| 3     | 0.206800      | 0.253372       | 0.704872  | 0.650684 | 0.676695 | 0.920898 |

### XLM-RoBERTa Model

| Epoch | Training Loss | Validation Loss | Precision | Recall   | F1       |
|-------|---------------|----------------|-----------|----------|----------|
| 1     | 0.323100      | 0.275503       | 0.775799  | 0.694886 | 0.733117 |
| 2     | 0.272500      | 0.262481       | 0.739266  | 0.739900 | 0.739583 |
| 3     | 0.248600      | 0.252498       | 0.751478  | 0.741152 | 0.746280 |
| 4     | 0.236800      | 0.249968       | 0.754882  | 0.741449 | 0.748105 |
| 5     | 0.223800      | 0.252187       | 0.764390  | 0.740460 | 0.752235 |
| 6     | 0.218600      | 0.249887       | 0.756352  | 0.741646 | 0.748927 |
| 7     | 0.209700      | 0.250748       | 0.760696  | 0.739438 | 0.749916 |

## Setup and Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/named-entity-recognition.git
   cd named-entity-recognition
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI app**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080
   ```

5. **Deploy on Fly.io**:
   Use the following steps to deploy the app on Fly.io.

## Fly.io Deployment

To deploy this FastAPI app on Fly.io, follow these steps:

### Step 1: Install Fly CLI
If you haven't already, install the Fly.io CLI:
```bash
curl -L https://fly.io/install.sh | sh
```

### Step 2: Authenticate with Fly.io
Log in to your Fly.io account:
```bash
fly auth login
```

### Step 3: Initialize Fly.io App
Run the following command in the root directory of your project:
```bash
fly launch
```
During the launch process:
- Fly will ask you for a unique app name.
- It will detect your `Dockerfile` automatically.
- Accept default region recommendations or specify your preferred region.

### Step 4: Scale Resources
Increase memory allocation for running the model. For example, to set the memory to 2 GB:
```bash
fly scale memory 2048
```

### Step 5: Deploy the App
Once configured, deploy the app with:
```bash
fly deploy
```

### Step 6: Monitor and Test
To check logs and ensure the app is running correctly:
```bash
fly logs
```

Access your deployed app at the Fly.io-provided URL (e.g., `https://your-app-name.fly.dev`).

## Usage

Access the web interface through the Fly.io URL or `http://localhost:8080` (if running locally) to test the NER model and view recognized entities.

This application leverages the XLM-RoBERTa model fine-tuned on Azerbaijani language data for high-accuracy named entity recognition.
