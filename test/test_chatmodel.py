import os

import zhipuai
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate

from chat_models import ChatGLM

_ = load_dotenv(find_dotenv())  # read local .env file

zhipuai.api_key = os.environ['CHATGLM_API_KEY']


def test_chatglm_model():
    prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
    model = ChatGLM(temperature=1)
    chain = prompt | model
    result = chain.invoke({"foo": "bears"})
    print(result)
