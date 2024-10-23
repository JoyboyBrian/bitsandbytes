import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_q4_0():
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="q4_0",
            # bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        logger.info("Loading model with Q4_0 quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            quantization_config=bnb_config,
            device_map="cpu",
            torch_dtype=torch.float16,
        )
        
        logger.info("Model loaded successfully")
        logger.info(f"Model device: {model.device}")
        
        logger.info("Creating tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        
        text = "I love China because"
        logger.info(f"Input text: {text}")
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        logger.info("Input tensor created")
        
        logger.info("Starting model.generate()...")
        try:
            outputs = model.generate(
                **inputs,
                max_length=50,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Generated text: {generated_text}")
            logger.info("Generation completed successfully")
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    test_q4_0()