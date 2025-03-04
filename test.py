# ---------------------------------------------------------------------------------------------------------------------------------------
from peft import LoraConfig, TaskType
from fate_client.pipeline.components.fate.nn.loader import LLMModelLoader

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
    target_modules=['query_key_value'],
)
lora_config.target_modules = list(lora_config.target_modules)  # Ensure JSON serializability

# Define the pretrained model path
pretrained_model_path = "fill with pretrained model download path please"

# Load the Deepseek model instead of ChatGLM
model = LLMModelLoader(
    "fed_utils.deepseek",  # Update to reflect DeepSeek
    "DeepseekV3ForCausalLM",  # Specify the correct model class
    pretrained_path=pretrained_model_path,
    peft_type="LoraConfig",
    peft_config=lora_config.to_dict(),
    trust_remote_code=True
)
