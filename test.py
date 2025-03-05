import pandas as pd
df = pd.read_json('AdvertiseGen/train.json', lines=True)

# ---------------------------------------------------------------------------------------------------------------------------------------
from peft import LoraConfig, TaskType
from fate_client.pipeline.components.fate.nn.loader import LLMModelLoader
from fate_client.pipeline.components.fate.nn.loader import LLMDatasetLoader, LLMDataFuncLoader

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
    target_modules=['query_key_value'],
)
lora_config.target_modules = list(lora_config.target_modules)  

# Define the pretrained model path
pretrained_model_path = "/media/sf_internship/DeepSeek-R1-Distill-Llama-8B"

# Load the Deepseek model instead of ChatGLM
model = LLMModelLoader(
    "fed_utils.deepseek",  # Update to reflect DeepSeek
    "DeepseekV3ForCausalLM",  # Specify the correct model class
    pretrained_path=pretrained_model_path,
    peft_type="LoraConfig",
    peft_config=lora_config.to_dict(),
    trust_remote_code=True
)

import time
from fate_client.pipeline.components.fate.reader import Reader
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate.homo_nn import HomoNN, get_config_of_seq2seq_runner
from fate_client.pipeline.components.fate.nn.algo_params import Seq2SeqTrainingArguments, FedAVGArguments
from fate_client.pipeline.components.fate.nn.loader import LLMModelLoader, LLMDatasetLoader, LLMDataFuncLoader
from peft import LoraConfig, TaskType


guest = '9999'
host = '9999'
arbiter = '9999'

epochs = 1
batch_size = 1
lr = 5e-4

ds_config = {
    "train_micro_batch_size_per_gpu": batch_size,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": lr,
            "torch_adam": True,
            "adam_w_mode": False
        }
    },
    "fp16": {
        "enabled": True
    },
    "gradient_accumulation_steps": 1,
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 1e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 1e8,
        "contiguous_gradients": True,
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        }
    }
}

pipeline = FateFlowPipeline().set_parties(guest=guest, host=host, arbiter=arbiter)
# pipeline.bind_local_path(path="", namespace="experiment", name="ad")
time.sleep(5)


reader_0 = Reader("reader_0", runtime_parties=dict(guest=guest, host=host))
reader_0.guest.task_parameters(
    namespace="experiment",
    name="ad"
)
reader_0.hosts[0].task_parameters(
    namespace="experiment",
    name="ad"
)


tokenizer_params = dict(
    tokenizer_name_or_path=pretrained_model_path,
    trust_remote_code=True,
)

dataset = LLMDatasetLoader(
    "prompt_dataset",
    "PromptDataset",
    **tokenizer_params,
)

data_collator = LLMDataFuncLoader(
    "cust_data_collator",
    "get_seq2seq_data_collator",
    **tokenizer_params,
)

conf = get_config_of_seq2seq_runner(
    algo='fedavg',
    model=model,
    dataset=dataset,
    data_collator=data_collator,
    training_args=Seq2SeqTrainingArguments(
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        remove_unused_columns=False, 
        predict_with_generate=False,
        deepspeed=ds_config,
        learning_rate=lr,
        use_cpu=False, # this must be set as we will gpu
        fp16=True,
    ),
    fed_args=FedAVGArguments(),
    task_type='causal_lm',
    save_trainable_weights_only=True # only save trainable weights
)

homo_nn_0 = HomoNN(
    'nn_0',
    runner_conf=conf,
    train_data=reader_0.outputs["output_data"],
    runner_module="homo_seq2seq_runner",
    runner_class="Seq2SeqRunner",
)

homo_nn_0.guest.conf.set("launcher_name", "deepspeed") # tell schedule engine to run task with deepspeed
homo_nn_0.hosts[0].conf.set("launcher_name", "deepspeed") # tell schedule engine to run task with deepspeed

pipeline.add_tasks([reader_0, homo_nn_0])
pipeline.conf.set("task", dict(engine_run={"cores": 1})) # the number of gpus of each party

pipeline.compile()
pipeline.fit()