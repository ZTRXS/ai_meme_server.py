from flask import Flask, request, jsonify, send_from_directory
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemeGenerator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the DialoGPT model"""
        try:
            logger.info("Loading DialoGPT model...")
            
            # Use smaller model for faster loading on free tier
            model_name = "microsoft/DialoGPT-small"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.is_loaded = True
            logger.info("✅ DialoGPT model loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            self.is_loaded = False

# Initialize the meme generator
meme_ai = MemeGenerator()

@app.route('/')
def home():
    return jsonify({
        "status": "British Meme AI Server",
        "model_loaded": meme_ai.is_loaded,
        "endpoints": {
            "generate_meme": "POST /api/generate-meme",
            "health": "GET /api/health"
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy" if meme_ai.is_loaded else "loading",
        "model_loaded": meme_ai.is_loaded
    })

@app.route('/api/generate-meme', methods=['POST'])
def generate_meme():
    try:
        if not meme_ai.is_loaded:
            return jsonify({
                "success": False,
                "error": "AI model still loading, please try again in 30 seconds"
            }), 503

        data = request.json
        topic = data.get('topic', 'British weather')
        style = data.get('style', 'sarcastic')
        
        logger.info(f"Generating meme about: {topic} with style: {style}")
        
        # Create prompt for British meme
        prompt = self.create_british_prompt(topic, style)
        
        # Generate with DialoGPT
        result = meme_ai.generator(
            prompt,
            max_length=150,
            num_return_sequences=1,
            temperature=0.9,
            do_sample=True,
            pad_token_id=meme_ai.tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
        
        generated_text = result[0]['generated_text']
        
        # Extract just the meme part (remove the prompt)
        meme_text = generated_text.replace(prompt, '').strip()
        
        # Clean up the response
        meme_text = self.clean_meme_text(meme_text)
        
        logger.info(f"Generated meme: {meme_text}")
        
        return jsonify({
            "success": True,
            "meme": meme_text,
            "topic": topic,
            "style": style,
            "model": "DialoGPT-small"
        })
        
    except Exception as e:
        logger.error(f"Error generating meme: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    def create_british_prompt(self, topic, style):
        """Create a proper prompt for British meme generation"""
        style_descriptions = {
            "sarcastic": "sarcastic and dry wit",
            "wholesome": "heartwarming and positive", 
            "roast": "playfully insulting and banter-filled",
            "posh": "upper-class and refined humor",
            "chav": "street-style British banter",
            "northern": "Northern English humor and slang"
        }
        
        style_desc = style_descriptions.get(style, "sarcastic and dry wit")
        
        prompt = f"""Create a funny British meme about {topic} in a {style_desc} style.

Meme:"""
        
        return prompt

    def clean_meme_text(self, text):
        """Clean up the generated meme text"""
        # Remove any incomplete sentences
        if '.' in text:
            text = text.split('.')[0] + '.'
        
        # Add emojis based on content
        if 'weather' in text.lower() or 'rain' in text.lower():
            text += ' ☔🌦️'
        elif 'tea' in text.lower() or 'cuppa' in text.lower():
            text += ' ☕🫖'
        elif 'football' in text.lower() or 'match' in text.lower():
            text += ' ⚽🏆'
        elif 'queue' in text.lower() or 'line' in text.lower():
            text += ' 📏🚶'
        else:
            text += ' 🇬🇧'
            
        return text

# Load model when server starts
@app.before_first_request
def load_model_on_start():
    meme_ai.load_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    # For Render.com, use 0.0.0.0
    app.run(host='0.0.0.0', port=port, debug=False)
