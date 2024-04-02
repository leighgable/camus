import os
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
)

READER_MODEL_NAME = "cognitivecomputations/dolphin-phi-2-kensho"
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
READER_MODEL_PATH = "./models/camus/"
EMBEDDING_MODEL_PATH = "./models/embedding/"
def download_model(model_path, model_name):
    """Download a Hugging Face model and tokenizer to the specified directory"""
    # Check if the directory already exists
    if not os.path.exists(model_path):
        # Create the directory
        os.makedirs(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Save the model and tokenizer to the specified directory
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

download_model("./models/camus", READER_MODEL_NAME)
download_model("./models/embedding", EMBEDDING_MODEL_NAME)
