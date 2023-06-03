import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global tokenizer

    model_name = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    # Run the model
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(inputs, max_length=100, no_repeat_ngram_size=2)
    result = tokenizer.decode(outputs[0])

    # Return the results as a dictionary
    return result
