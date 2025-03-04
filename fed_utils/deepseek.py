# TODO: insert copyright thingies
from fed_utils.PELLM import PELLM
from transformers import AutoConfig
from DeepSeek.models.deepseek import DeepseekV3ForCausalLM, DeepseekV3Config


class DeepSeek(PELLM):
    config_class = DeepseekV3Config

    def __init__(self,
                 pretrained_path: str = None,
                 peft_type: str = None,
                 peft_config: dict = None,
                 **kwargs) -> None:

        super().__init__(pretrained_path=pretrained_path,
                         peft_type=peft_type,
                         peft_config=peft_config,
                         **kwargs)

    def init_base_lm(self, **kwargs):
        if self.config is not None:
            self._pe_lm = DeepseekV3ForCausalLM.from_pretrained(self.config_path,
                                                           config=self.config,
                                                           torch_dtype=self.torch_dtype,
                                                           **kwargs)
        elif self.config_path is not None:
            self._pe_lm = DeepseekV3ForCausalLM.from_pretrained(self.config_path, torch_dtype=self.torch_dtype, **kwargs)
        else:
            raise ValueError(
                'config_path to pretrained model folder cannot be None')

    def check_config(self, pretrain_path):
        config = AutoConfig.from_pretrained(pretrain_path)
        assert isinstance(
            config, DeepseekV3Config), 'The config of pretrained model must be DeepseekV3Config, but got {}'.format(
            type(config))
