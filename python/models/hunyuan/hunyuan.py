import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type
from urllib.parse import urlparse
from datetime import datetime 
from base import *
from tencentapi import _signatureV3
import requests
from dataclasses import dataclass 

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://hunyuan.tencentcloudapi.com"
DEFAULT_PATH = "/"


# def _convert_message_to_dict(message: BaseMessage) -> dict:
#     message_dict: Dict[str, Any]
#     if isinstance(message, ChatMessage):
#         message_dict = {"role": message.role, "content": message.content}
#     elif isinstance(message, HumanMessage):
#         message_dict = {"role": "user", "content": message.content}
#     elif isinstance(message, AIMessage):
#         message_dict = {"role": "assistant", "content": message.content}
#     else:
#         raise TypeError(f"Got unknown type {message}")

#     return message_dict

def _convert_message_to_dict(message: ChatMessage) -> dict:
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"Role": message.role, "Content": message.content}
    else:
        raise TypeError(f"Got unknown type {message}")

    return message_dict

# def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
#     role = _dict["role"]
#     if role == "user":
#         return HumanMessage(content=_dict["content"])
#     elif role == "assistant":
#         return AIMessage(content=_dict.get("content", "") or "")
#     else:
#         return ChatMessage(content=_dict["content"], role=role)

def _convert_dict_to_message(_dict: Mapping[str, Any]) -> ChatMessage:
    role = _dict["Role"]

    return ChatMessage(content=_dict["Content"], role=role)
    

def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[DeltaMessage]
) -> DeltaMessage:
    role = _dict.get("Role")
    content = _dict.get("Content") or ""


    return DeltaMessage(content=content, role=role)


# def _create_chat_result(response: Mapping[str, Any]) -> ChatCompletionResponse:
#     generations = []
#     for choice in response["choices"]:
#         message = _convert_dict_to_message(choice["messages"])
#         generations.append(ChatCompletionResponse(message=message))

#     token_usage = response["usage"]
#     llm_output = {"token_usage": token_usage}
#     return ChatResult(generations=generations, llm_output=llm_output)

@dataclass
class ChatHunyuan():
    """Tencent Hunyuan chat models API by Tencent.
    For more information, see https://cloud.tencent.com/document/product/1729
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "hunyuan_app_id": "HUNYUAN_APP_ID",
            "hunyuan_secret_id": "HUNYUAN_SECRET_ID",
            "hunyuan_secret_key": "HUNYUAN_SECRET_KEY",
        }

    @property
    def lc_serializable(self) -> bool:
        return True

    hunyuan_api_base: str = DEFAULT_API_BASE
    """Hunyuan custom endpoints"""
    hunyuan_app_id: Optional[int] = None
    """Hunyuan App ID"""
    hunyuan_secret_id: Optional[str] = None
    """Hunyuan Secret ID"""
    hunyuan_secret_key: Optional[str] = None
    """Hunyuan Secret Key"""
    streaming: bool = True
    """Whether to stream the results or not."""
    request_timeout: int = 60
    """Timeout for requests to Hunyuan API. Default is 60 seconds."""

    query_id: Optional[str] = None
    """Query id for troubleshooting"""
    temperature: float = 1.0
    """What sampling temperature to use."""
    top_p: float = 1.0
    """What probability mass to use."""
    model: str = "ChatStd"

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for API call not explicitly specified."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Hunyuan API."""
        normal_params = {
            "app_id": self.hunyuan_app_id,
            "secret_id": self.hunyuan_secret_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "model": self.model,
        }

        if self.query_id is not None:
            normal_params["query_id"] = self.query_id
        return {**normal_params, **self.model_kwargs.default_factory()}


    # def _generate(
    #     self,
    #     messages: List[ChatMessage],
    #     stop: Optional[List[str]] = None,
    #     **kwargs: Any,
    # ) -> ChatCompletionResponse:
    #     if self.streaming:
    #         stream_iter = self._stream(
    #             messages=messages, stop=stop,  **kwargs
    #         )
    #         return generate_from_stream(stream_iter)

    #     res = self._chat(messages, **kwargs)

    #     response = res.json()

    #     if "error" in response:
    #         raise ValueError(f"Error from Hunyuan api response: {response}")

    #     return _create_chat_result(response)

    def _stream(
        self,
        messages: List[ChatMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[DeltaMessage]:
        res = self._chat(messages, **kwargs)

        default_chunk_class = DeltaMessage
        for chunk in res.iter_lines():
            if chunk == b"":  # Skip empty lines
                continue
            chunkstr = chunk.decode("utf-8")

            # If the chunk starts with "data:", remove it
            if(chunkstr.startswith("data:")) :
                chunkstr = chunkstr[5:]

            response = json.loads(chunkstr)
            if "error" in response:
                raise ValueError(f"Error from Hunyuan api response: {response}")

            for choice in response["Choices"]:
                delta = _convert_delta_to_message_chunk(
                    choice["Delta"], default_chunk_class
                )
                # default_chunk_class = chunk.__class__
                yield delta 
                # if run_manager:
                #     run_manager.on_llm_new_token(chunk.content)

    def _chat(self, messages: List[ChatMessage], **kwargs: Any) -> requests.Response:
        if self.hunyuan_secret_key is None:
            raise ValueError("Hunyuan secret key is not set.")
        parameters = {**self._default_params, **kwargs}

        headers = parameters.pop("headers", {})
        timestamp = parameters.pop("timestamp", int(time.time()))
        expired = parameters.pop("expired", timestamp + 24 * 60 * 60)

        # payload = {
        #     "timestamp": timestamp,
        #     "expired": expired,
        #     "Messages": [_convert_message_to_dict(m) for m in messages],
        #     **parameters,
        # }
        top_p = parameters.get("top_p", 1.0)
        temperature = parameters.get("temperature", 1.0)
        payload = {
            "TopP": top_p,
            "Temperature": temperature,
            "Messages": [_convert_message_to_dict(m) for m in messages],
        }
        # if self.streaming:
        #     payload["stream"] = 1

        url = self.hunyuan_api_base + DEFAULT_PATH
        res = requests.post(
            url=url,
            timeout=self.request_timeout,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                 "Accept-Charset": "UTF-8",
                # "Authorization": _signature(
                #     secret_key=self.hunyuan_secret_key, url=url, payload=payload
                # ),
                "Authorization": _signatureV3(
                    secret_id=self.hunyuan_secret_id,
                    secret_key=self.hunyuan_secret_key,
                    action=self.model,
                    timestamp=str(timestamp),
                    payload=json.dumps(payload)
                ),
                "X-TC-Action": self.model,
                "X-TC-Version": "2023-09-01",
                # "X-TC-Region": "ap-guangzhou",
                "X-TC-Timestamp": str(timestamp) ,

                **headers,
            },
            json=payload,
            stream=self.streaming,
        )
        return res

    @property
    def _llm_type(self) -> str:
        return "hunyuan-chat-new"
