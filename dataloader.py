from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.vectorstores.chroma import Chroma
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from bs4 import BeautifulSoup as Soup
from langchain.utils.html import (PREFIXES_TO_IGNORE_REGEX,
                                  SUFFIXES_TO_IGNORE_REGEX)

from config import *

import logging
import sys
import shutil

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def init_index():
    if not INIT_INDEX:
        logging.info("continue without initializing index")
        return
    else:
        try:
            shutil.rmtree(INDEX_PERSIST_DIRECTORY, onerror=None)
        except OSError:
            print("Database already removed")

    # scrape data from web
    url_loader = RecursiveUrlLoader(
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
    )

    # load data from files
    file_loader = DirectoryLoader(TARGET_DIRECTORY)

    # merge data from URL and files
    all_loaders = MergedDataLoader(loaders=[url_loader, file_loader])

    document = all_loaders.load()

    # create embeddings with huggingface embedding model `all-MiniLM-L6-v2`
    # then persist the vector index on vector db
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    logging.info("index creating with `%d` documents", len(document))

    CHROMA_SETTINGS = Settings(
        anonymized_telemetry=True,
        is_persistent=True
    )
    
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