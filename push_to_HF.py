import os
from dotenv import load_dotenv
from huggingface_hub import login, HfApi

# Load the Hugging Face token from .env
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Log in to Hugging Face
login(token=hf_token)

# Define your repository ID
repo_id = "IsmatS/mbert-az-ner"

# Initialize HfApi and upload the model folder
api = HfApi()
api.upload_folder(folder_path="./mbert-azerbaijani-ner", path_in_repo="", repo_id=repo_id)
