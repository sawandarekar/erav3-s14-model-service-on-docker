import gradio as gr
import requests
import json
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get backend URL from environment variable, default to localhost if not set
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8000')

def process_text(text, max_length):
    if not text:
        return "Please enter some text before submitting"
    
    logger.info(f"requesting received at Frontend for text: {text} and max_length: {max_length}")
    try:
        # Call backend API using environment variable
        response = requests.post(
            f"{BACKEND_URL}/process",
            json={
                "text": text,
                "max_length": max_length
            }
        )
        # logger.info("response : ", response.json())
        # logger.info("response status code : ", response.status_code)
        if response.status_code == 200:
            result = response.json()
            return json.dumps(result.get("data").get("generated_text"), indent=2)
        else:
            return f"Error: {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        return f"Error connecting to backend service: {str(e)}"

# Create the Gradio interface
def create_app():
    interface = gr.Interface(
        fn=process_text,
        inputs=[
            gr.Textbox(
                lines=8,
                placeholder="Enter your text here...",
                label="Input Text",
                max_lines=20,
            ),
            gr.Slider(
                minimum=1,
                maximum=100,
                value=20,
                step=1,
                label="Max Token Length",
                info="Slide to adjust the maximum token length (100-1000)"
            )
        ],
        outputs=[
            gr.Textbox(
                label="Generated Text",
                lines=8,
            )
        ],
        title="SmolLM2 Text Generation - As Docker Container",
        description="Enter your text and adjust the max token length using the slider.",
        theme=gr.themes.Soft(),
        css="footer {display: none !important;}"
    )
    return interface

if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860) 