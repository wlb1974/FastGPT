# coding=utf-8
import argparse
import time
from typing import List, Literal, Optional, Union
import json
import numpy as np
import tiktoken
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from starlette.status import HTTP_401_UNAUTHORIZED
from pydantic import BaseModel, Field
from base import *
from hunyuan import ChatHunyuan
from embedding import HunyuanEmbedding
import logging
import os 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




async def verify_token(request: Request):
    auth_header = request.headers.get('Authorization')
    if auth_header:
        token_type, _, token = auth_header.partition(' ')
        if (
            token_type.lower() == "bearer"
            and token == "sk-aaabbbcccdddeeefffggghhhiiijjjkkk"
        ):  # 这里配置你的token
            return True
    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid authorization credentials",
    )


class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens


# def expand_features(embedding, target_length):
#     poly = PolynomialFeatures(degree=2)
#     expanded_embedding = poly.fit_transform(embedding.reshape(1, -1))
#     expanded_embedding = expanded_embedding.flatten()
#     if len(expanded_embedding) > target_length:
#         # 如果扩展后的特征超过目标长度，可以通过截断或其他方法来减少维度
#         expanded_embedding = expanded_embedding[:target_length]
#     elif len(expanded_embedding) < target_length:
#         # 如果扩展后的特征少于目标长度，可以通过填充或其他方法来增加维度
#         expanded_embedding = np.pad(
#             expanded_embedding, (0, target_length - len(expanded_embedding))
#         )
#     return expanded_embedding


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest
):
    global model, tokenizer

    logging.info(f"completions request: {request}")
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    query = request.messages[-1].content

    # prev_messages = request.messages[:-1]
    # if len(prev_messages) > 0 and prev_messages[0].role == "system":
    #     query = prev_messages.pop(0).content + query

    history = request.messages[:-1]
    # if len(prev_messages) % 2 == 0:
    #     for i in range(0, len(prev_messages), 2):
    #         if (
    #             prev_messages[i].role == "user"
    #             and prev_messages[i + 1].role == "assistant"
    #         ):
    #             history.append([prev_messages[i].content, prev_messages[i + 1].content])

    if request.stream:
        generate = predict(query, history, request.model)
        return EventSourceResponse(generate, media_type="text/event-stream")

    response, _ = model.chat(tokenizer, query, history=history)
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop",
    )

    return ChatCompletionResponse(
        model=request.model, choices=[choice_data], object="chat.completion"
    )


async def predict(query: str, history: List[List[str]], model_id: str):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(json.dumps(chunk.to_dict(), ensure_ascii=False))

    message = history
    message.append(ChatMessage(role="user", content=query))

    for new_response in model._stream(messages=history, model = model_id):
        delta = DeltaMessage(role="assistant", content=new_response.content)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=delta, finish_reason=None
        )
        chunk = ChatCompletionResponse(
            model=model_id, choices=[choice_data], object="chat.completion.chunk"
        )
        yield "{}".format(json.dumps(chunk.to_dict(),ensure_ascii=False))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(), finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(json.dumps(chunk.to_dict(),  ensure_ascii=False))
    yield '[DONE]'


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(
    request: EmbeddingRequest
):
    global hunyuanEmbedding

    logging.info(f"embeddings request: {request}")

    embeddings = hunyuanEmbedding.embed_documents(request.input)

    logging.info(f"humyuan embeddings: {embeddings}")
    
    # 将numpy数组转换为列表
    # embeddings = [embedding for embedding in embeddings]
    prompt_tokens = sum(len(text.split()) for text in request.input)
    total_tokens = sum(num_tokens_from_string(text) for text in request.input)

    response = {
        "data": [
            {"embedding": embedding, "index": index, "object": "embedding"}
            for index, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "object": "list",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        },
    }

    return response


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", default="16", type=str, help="Model name")
    # args = parser.parse_args()

    # model_dict = {
    #     "4": "THUDM/chatglm2-6b-int4",
    #     "8": "THUDM/chatglm2-6b-int8",
    #     "16": "THUDM/chatglm2-6b",
    # }

    # model_name = model_dict.get(args.model_name, "THUDM/chatglm2-6b")

    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModel.from_pretrained(model_name, trust_remote_code=True).cuda()
    # embeddings_model = SentenceTransformer('moka-ai/m3e-large', device='cpu')
    
    logging.basicConfig(level=logging.INFO)

    model = ChatHunyuan(    
        hunyuan_app_id=os.getenv("HUNYUAN_APP_ID"),
        hunyuan_secret_id=os.getenv("HUNYUAN_SECRET_ID"),
        hunyuan_secret_key=os.getenv("HUNYUAN_SECRET_KEY"),
        model="ChatPro",
        temperature=0)

    hunyuanEmbedding = HunyuanEmbedding(
        hunyuan_app_id=os.getenv("HUNYUAN_APP_ID"),
        hunyuan_secret_id=os.getenv("HUNYUAN_SECRET_ID"),
        hunyuan_secret_key=os.getenv("HUNYUAN_SECRET_KEY"),
    )
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)
