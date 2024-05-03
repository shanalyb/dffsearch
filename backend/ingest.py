import logging
import os

from langchain.document_loaders import GithubFileLoader
from langchain.indexes import index
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.indexes import SQLRecordManager, index
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
FOLDER_ID = os.getenv("FOLDER_ID")


def get_embeddings_model() -> Embeddings:
    return YandexGPTEmbeddings(folder_id=FOLDER_ID, api_key=os.getenv("API_KEY"), sleep_interval=0.1)


def load_dff_docs():
    return GithubFileLoader(
        repo="deeppavlov/dialog_flow_framework",
        branch="master",
        access_token=os.getenv("GITHUB_ACCESS_TOKEN"),
        github_api_url="https://api.github.com",
        file_filter=lambda file_path: file_path.endswith(".md") or file_path.endswith(".rst") or file_path.endswith(".py"),  
    ).load()

def ingest_docs():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    embedding = get_embeddings_model()

    docs_from_documentation = load_dff_docs()

    docs_transformed = text_splitter.split_documents(docs_from_documentation)

    persist_directory = "./chromadb.db"
    vectorstore = Chroma.from_documents(docs_transformed, embedding, persist_directory=persist_directory)

    namespace = "chromadb/my_documents"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    logger.info(f"Indexing stats: {indexing_stats}")


if __name__ == "__main__":
    ingest_docs()
