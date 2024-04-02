from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
import time


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

    def to_dict(self):
        return {
            'role': self.role,
            'content': self.content
        }


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None

    def to_dict(self):
        return {
            'role': self.role,
            'content': self.content
        }


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False

    def to_dict(self):
        return {
            'model': self.model,
            'messages': [message.to_dict() for message in self.messages],
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_length': self.max_length,
            'stream': self.stream
        }


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]

    def to_dict(self):
        return {
            'index': self.index,
            'message': self.message.dict(),
            'finish_reason': self.finish_reason
        }


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]

    def to_dict(self):
        return {
            'index': self.index,
            'delta': self.delta.dict(),
            'finish_reason': self.finish_reason
        }


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[
        Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
    ]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))

    def to_dict(self):
        return {
            'model': self.model,
            'choices': [choice.to_dict() for choice in self.choices],
            'object': self.object,
            'created': self.created
        }
