from typing import List, Optional, Any, Dict, Mapping

import zhipuai
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.pydantic_v1 import Field


def convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any]
    if isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict.get("role")
    if role == 'assistant':
        return AIMessage(content=_dict.get("content", ""), additional_kwargs={})
    raise TypeError(f"Got unknown type {_dict}")


class ChatGLM(BaseChatModel):
    model_name: str = "chatglm_turbo"
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    temperature: float = 0.95
    top_p: float = 0.7
    return_type: str = "text"

    @property
    def _default_params(self) -> Dict[str, Any]:
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "return_type": self.return_type
        }
        return params

    def _create_chat_result(self, response: dict) -> ChatResult:
        generations = []
        data = response["data"]
        for choice in data["choices"]:
            message = convert_dict_to_message(choice)
            generation_info = {"finish_reason": "stop"}
            gen = ChatGeneration(
                message=message,
                generaion_info=generation_info,
            )
            generations.append(gen)
        token_usage = data.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        message_dicts = [convert_message_to_dict(m) for m in messages]
        params = {
            **self.model_kwargs,
            **self._default_params,
            **kwargs
        }
        response = zhipuai.model_api.invoke(
            model=self.model_name,
            prompt=message_dicts,
            temperature=params["temperature"],
            top_p=params["top_p"],
            return_type=params["return_type"],
        )
        return self._create_chat_result(response)

    @property
    def _llm_type(self) -> str:
        return "chat-glm-wrapper"
