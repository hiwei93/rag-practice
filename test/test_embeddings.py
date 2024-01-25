import os

import zhipuai
from dotenv import load_dotenv, find_dotenv

from embeddings.chatglm import ChatGLMEmbeddings

_ = load_dotenv(find_dotenv())  # read local .env file

zhipuai_api_key = os.environ['CHATGLM_API_KEY']


def test_chatglm_embeddings():
    embeddings = ChatGLMEmbeddings(
        zhipuai_api_key=zhipuai_api_key
    )
    print(embeddings.embed_documents(["你好"]))

