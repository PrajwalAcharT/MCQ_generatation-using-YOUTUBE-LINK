from haystack.nodes import PromptModelInvocationLayer
from llama_cpp import Llama
import os
from typing import Dict, List, Union, Optional

class LlamaCPPInvocationLayer(PromptModelInvocationLayer):
    def __init__(self, model_name_or_path: str,
                 max_length: int = 128,
                 max_context: int = 2048,
                 use_gpu: bool = False):
        
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.model = Llama(
            model_path=model_name_or_path,
            n_ctx=max_context,
            n_threads=4,
            verbose=False
        )

    def invoke(self, prompt: str, **kwargs) -> List[str]:
        stream = kwargs.pop("stream", False)
        max_tokens = kwargs.get("max_tokens", self.max_length)
        
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            echo=False,
            temperature=0.7,
            top_p=0.95,
        )
        
        return [o['text'] for o in output['choices']]

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        return True