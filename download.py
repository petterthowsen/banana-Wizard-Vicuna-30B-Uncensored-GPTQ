# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface GPTJ model

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    print("downloading model...")
    model = AutoModelForCausalLM.from_pretrained("TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ")
    print("done")

    print("downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ")
    print("done")

if __name__ == "__main__":
    download_model()