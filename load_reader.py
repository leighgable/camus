from transformers import pipeline
import torch
from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, 
                          BitsAndBytesConfig, 
                          Pipeline
)
from download_models import READER_MODEL_PATH, EMBEDDING_MODEL_PATH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # bnb_4bit_use_double_quant=True, # crazy kensho architecture doesn't support
    bnb_4bit_quant_type="nf4",        # double quant.
    bnb_4bit_compute_dtype=torch.bfloat16,
) if (device == "cuda") or (device == "mps") else None 

model = AutoModelForCausalLM.from_pretrained(READER_MODEL_PATH, 
                                             quantization_config=bnb_config, 
                                             device_map="auto", 
                                             trust_remote_code=True
                                             )
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)

READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation", # change to conversational
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=512,
)
