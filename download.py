# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model_name = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

if __name__ == "__main__":
    download_model()
