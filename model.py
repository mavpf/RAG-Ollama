from langchain_community.llms import Ollama
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from bs4 import BeautifulSoup as Soup
from langchain.utils.html import (PREFIXES_TO_IGNORE_REGEX,
                                  SUFFIXES_TO_IGNORE_REGEX)

from config import *

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


global conversation
conversation = None



def init_conversation():
    global convo_chain

    # load index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(collection_name='v_db',embedding_function=embeddings,persist_directory=INDEX_PERSIST_DIRECTORY)

    # llama2 llm which runs with ollama
    # ollama expose an api for the llam in `localhost:11434`
    llm = Ollama(
        model="llama2",
        base_url="http://localhost:11434",
        verbose=True,
    )

    # create prompt with LCEL

    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONDENSED_QUESTION),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, vectordb.as_retriever(), condense_question_prompt
    )


    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ]
    )

    chain = create_stuff_documents_chain(llm, prompt)

    convo_chain = create_retrieval_chain(history_aware_retriever, chain)


def chat(question):
    global convo_chain

    chat_history = []
    response = convo_chain.invoke({"input": question, "chat_history": chat_history})
    answer = response['answer']

    logging.info("got response from llm - %s", answer)

    return answer
