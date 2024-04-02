from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime 
import hashlib
import hmac
import requests
import json
import time
from tencentapi import _signatureV3
from base import *
from dataclasses import dataclass 


logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://hunyuan.tencentcloudapi.com"
DEFAULT_PATH = "/"

@dataclass
class HunyuanEmbedding():
    """`Tencent Hunyuan Embeddings` embedding models."""

    hunyuan_api_base: str = DEFAULT_API_BASE
    """Hunyuan custom endpoints"""
    hunyuan_app_id: Optional[int] = None
    """Hunyuan App ID"""
    hunyuan_secret_id: Optional[str] = None
    """Hunyuan Secret ID"""
    hunyuan_secret_key: Optional[str] = None
    """Hunyuan Secret Key"""
    request_timeout: int = 60
    """Timeout for requests to Hunyuan API. Default is 60 seconds."""


    chunk_size: int = 16
    """Chunk size when multiple texts are input"""

    model: str = "GetEmbedding"
    """Model name
    you could get from https://cloud.tencent.com/document/api/1729/102832
       
    preset models are mapping to an endpoint.
    `model` will be ignored if `endpoint` is set
    """

    init_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """init kwargs for hunyuan client init, such as `query_per_second` which is 
        associated with hunyuan resource object to limit QPS"""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """extra params for model invoke using with `do`."""

   
    def embed_query(self, text: str) -> List[float]:
        resp = self.embed_documents([text])
        return resp[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of text documents using the AutoVOT algorithm.

        Args:
            texts (List[str]): A list of text documents to embed.

        Returns:
            List[List[float]]: A list of embeddings for each document in the input list.
                            Each embedding is represented as a list of float values.
        """
        # text_in_chunks = [
        #     texts[i : i + self.chunk_size]
        #     for i in range(0, len(texts), self.chunk_size)
        # ]
        lst = []
        for chunk in texts:
            # print(f"chunk: {chunk}")
            resp = self.hunyuan_embedding_text(chunk)
            embed = []
            for res in resp["Data"]:
                # Convert each float to a string
                # embedding_as_strings = list(map(str, res["Embedding"]))
                embed.extend(res["Embedding"])
            lst.append(embed)
        return lst

    async def aembed_query(self, text: str) -> List[float]:
        # embeddings = self.aembed_documents([text])
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)
        # text_in_chunks = [
        #     texts[i : i + self.chunk_size]
        #     for i in range(0, len(texts), self.chunk_size)
        # ]
        # lst = []
        # for chunk in text_in_chunks:
        #     resp = self.ahunyuan_embedding_text(chunk)
        #     for res in resp["Data"]:
        #         lst.extend(res["Embedding"])
        # return lst

    async def ahunyuan_embedding_text(self, text : str) -> List[float]:
        return await self.hunyuan_embedding_text(text)
    
    def hunyuan_embedding_text(self, text : str) -> List[float]:
        if self.hunyuan_secret_key is None:
            raise ValueError("Hunyuan secret key is not set.")
        timestamp = int(time.time())
        payload = {
            "Input": f"{text}",
        }

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
                    action="GetEmbedding",
                    timestamp=str(timestamp),
                    payload=json.dumps(payload)
                ),
                "X-TC-Action": "GetEmbedding",
                "X-TC-Version": "2023-09-01",
                # "X-TC-Region": "ap-guangzhou",
                "X-TC-Timestamp": str(timestamp) ,
            },
            json=payload,
        )
        result = res.json()
        return result.get("Response", {})
