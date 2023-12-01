import runpod
from query_influence import query_influence
from dataset import train_dataset # OpenWebText
from model import GPT, GPTConfig
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT().to(device)

# TODO init OpenWebText dataset subset


# Load ihvp into memory
model_name = model._get_name()
num_params = "%.2fM" % (model.get_num_params()/1e6,)
if os.path.exists(f'minGPT_ihvp_v1.pt'):
    ihvp = torch.load(f'minGPT_ihvp_v1.pt')
    ihvp_tokenwise = torch.load(f'minGPT_ihvp_tokenwise_v1.pt')

def handler(job):
    job_input = job["input"] # Access the input from the request.
    query = job_input['query']
    
    # Pass input query into query_influence and return result
    result = query_influence([query], train_dataset, model, ihvp, ihvp_tokenwise)
    
    return result

runpod.serverless.start({ "handler": handler}) # Required.
