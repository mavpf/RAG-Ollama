import os

# define init index
INIT_INDEX = os.getenv('INIT_INDEX', 'false').lower() == 'true'

# vector index persist directory
INDEX_PERSIST_DIRECTORY = os.getenv('INDEX_PERSIST_DIRECTORY', "./data/chromadb")

# target url to scrape
TARGET_URL =  os.getenv('TARGET_URL', "https://python.langchain.com/v0.2/docs/concepts/")
DEPTH = 2

# target directory for file update
TARGET_DIRECTORY = os.getenv('TARGET_DIRECTORY', "./files")

# http api port
HTTP_PORT = os.getenv('HTTP_PORT', 7654)

# context

CONDENSED_QUESTION = os.getenv('CONDENSED_QUESTION', "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is.")

# prompt

SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}")
