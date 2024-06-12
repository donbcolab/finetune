import subprocess
import os
import torch
from transformers import (
    BitsAndBytesConfig,
    AutoConfig,
    AutoModelForCausalLM,
    LlavaNextForConditionalGeneration,
    LlavaForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    Idefics2ForConditionalGeneration,
)
import spaces
from joblib import Memory

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Create a cache directory if it doesn't exist
cache_dir = ".cache"
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cache_dir, verbose=0)

# Install required package
def install_flash_attn():
    subprocess.run(
        "pip install flash-attn --no-build-isolation",
        env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
        shell=True,
    )

# Architecture to model class mapping
ARCHITECTURE_MAP = {
    "LlavaNextForConditionalGeneration": LlavaNextForConditionalGeneration,
    "LlavaForConditionalGeneration": LlavaForConditionalGeneration,
    "PaliGemmaForConditionalGeneration": PaliGemmaForConditionalGeneration,
    "Idefics2ForConditionalGeneration": Idefics2ForConditionalGeneration,
    "AutoModelForCausalLM": AutoModelForCausalLM,
}

# Function to get the model summary with caching and GPU support
@spaces.GPU 
@memory.cache
def get_model_summary(model_name):
    """
    Retrieve the model summary for the given model name, handling quantization and offloading to GPU.

    Args:
        model_name (str): The name of the model to retrieve the summary for.

    Returns:
        tuple: A tuple containing the model summary (str) and an error message (str), if any.
    """
    try:
        # Fetch the model configuration
        config = AutoConfig.from_pretrained(model_name)
        architecture = config.architectures[0]
        quantization_config = getattr(config, "quantization_config", None)

        # Set up BitsAndBytesConfig if the model is quantized
        if quantization_config:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quantization_config.get("load_in_4bit", False),
                load_in_8bit=quantization_config.get("load_in_8bit", False),
                bnb_4bit_compute_dtype=quantization_config.get(
                    "bnb_4bit_compute_dtype", torch.float16
                ),
                bnb_4bit_quant_type=quantization_config.get(
                    "bnb_4bit_quant_type", "nf4"
                ),
                bnb_4bit_use_double_quant=quantization_config.get(
                    "bnb_4bit_use_double_quant", False
                ),
                llm_int8_enable_fp32_cpu_offload=quantization_config.get(
                    "llm_int8_enable_fp32_cpu_offload", False
                ),
                llm_int8_has_fp16_weight=quantization_config.get(
                    "llm_int8_has_fp16_weight", False
                ),
                llm_int8_skip_modules=quantization_config.get(
                    "llm_int8_skip_modules", None
                ),
                llm_int8_threshold=quantization_config.get(
                    "llm_int8_threshold", 6.0
                ),
            )
        else:
            bnb_config = None

        # Get the appropriate model class from the architecture map
        model_class = ARCHITECTURE_MAP.get(architecture, AutoModelForCausalLM)

        # Load the model, enabling trust_remote_code for Llama models
        model = model_class.from_pretrained(
            model_name,
            config=config,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",  # Use 'auto' for automatic device placement
        )

        model_summary = str(model) if model else "Model architecture not found."
        return model_summary, ""
    except ValueError as ve:
        return "", f"ValueError: {ve}"
    except EnvironmentError as ee:
        return "", f"EnvironmentError: {ee}"
    except Exception as e:
        return "", str(e)


def get_available_options(key, config):
    """
    Retrieve a list of available options from the configuration file.

    Args:
        key (str): The key in the configuration file.
        config (dict): The configuration dictionary.

    Returns:
        list: A list of available options.
    """
    options = config.get(key, {})
    return list(options.keys())


def format_metrics_output(metrics):
    """
    Format the metrics dictionary for output.

    Args:
        metrics (dict): A dictionary of evaluation metrics.

    Returns:
        str: A formatted string containing the metrics.
    """
    return "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())  # Format to 4 decimal places