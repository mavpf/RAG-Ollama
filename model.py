from langchain_community.llms import Ollama
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.vectorstores.chroma import Chroma
from chromadb.config import Settings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from bs4 import BeautifulSoup as Soup
from langchain.utils.html import (PREFIXES_TO_IGNORE_REGEX,
                                  SUFFIXES_TO_IGNORE_REGEX)

from config import *
import shutil
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


global conversation
conversation = None


def init_index():
    if not INIT_INDEX:
        logging.info("continue without initializing index")
        return
    else:
        shutil.rmtree(INDEX_PERSIST_DIRECTORY, onerror=None)

    # scrape data from web
    document = RecursiveUrlLoader(
        TARGET_URL,
        max_depth= DEPTH,
        extractor=lambda x: Soup(x, "html.parser").text,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        check_response_status=True,
        # drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
    ).load()

    logging.info("index creating with `%d` documents", len(document))

    CHROMA_SETTINGS = Settings(
        anonymized_telemetry=True,
        is_persistent=True
    )

    # create embeddings with huggingface embedding model `all-MiniLM-L6-v2`
    # then persist the vector index on vector db
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    logging.info("Init DB")

    try:
        vectordb = Chroma.from_documents(
            collection_name="v_db",
            documents=document,
            persist_directory=INDEX_PERSIST_DIRECTORY,
            embedding=embeddings,
            client_settings=CHROMA_SETTINGS
        )
        vectordb.persist()
    except:
        logging.error("Failure in DB creation")
    else:
        # if DB is loaded, then stop scrap
        os.environ["INIT_INDEX"]="false"
    

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
