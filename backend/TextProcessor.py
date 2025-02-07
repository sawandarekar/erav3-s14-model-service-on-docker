import torch
from model import LlamaForCausalLM
from transformers import AutoTokenizer
import yaml
import os

def load_config():
    with open('SmolLM2-135.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_model(config, checkpoint_path):
    model = LlamaForCausalLM(config)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Debug information
        # print("Model state keys:", model.state_dict().keys())
        # print("Checkpoint state keys:", checkpoint['model_state_dict'].keys())
        
        try:
            # Try strict loading first
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Model loaded successfully!")
        except Exception as e:
            # print(f"Error loading model: {str(e)}")
            
            # Try to map the state dict keys
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            
            for key in state_dict:
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix
                else:
                    new_key = key
                new_state_dict[new_key] = state_dict[key]
            
            # Try loading with the modified state dict
            model.load_state_dict(new_state_dict, strict=False)
            print("Model loaded with key remapping!")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
    
    model.eval()
    return model

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def generate_text(prompt, max_length=100):
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text
    with torch.no_grad():
        model.eval()
        outputs = model.generate(input_ids, max_length=max_length)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    return generated_text

# Load configuration and model
config = load_config()
device = get_device()
checkpoint_path = "checkpoint_5070.pt"
model = load_model(config, checkpoint_path).to(device)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")