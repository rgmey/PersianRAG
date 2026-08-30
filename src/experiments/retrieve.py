from typing import Iterable, List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from utils.decorators import timer
from langchain_community.vectorstores import FAISS
import logging
from hazm import Normalizer
from warnings import filterwarnings
filterwarnings('ignore')


class Retriever:
    def __init__(self, 
                 new_vdb=True, 
                #  embedding_model=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small",
                 embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                                       model_kwargs={"local_files_only": True})
                ) -> None:
        self.new_vdb = new_vdb
        self.vdb = None
        self.retrieved_docs = None
        self.is_persian = True
        self.embedding_model=embedding_model
        self.final_docs = ""

    @timer
    def hazmer(self, raw_texts):
        normalizer = Normalizer()
        normalized_raw_texts = [normalizer.normalize(text) for text in raw_texts]
        return normalized_raw_texts
    
    @timer
    def create_vdb(self,
        raw_texts: Iterable[str],
        chunk_size: int,
        chunk_overlap: int,
        vdb_dir: str = "faiss_vdb",) -> None:
        if not raw_texts:
            raise ValueError("raw_texts must be a non-empty iterable of strings")

        if not self.new_vdb:
            try:
                self.vdb = FAISS.load_local(
                    folder_path=vdb_dir,
                    embeddings=self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                logging.getLogger(__name__).debug("Reusing existing vector DB at %s", vdb_dir)
                return
            except Exception:  # fallback to rebuilding if load fails
                logging.getLogger(__name__).warning(
                    "Failed to load existing VDB; rebuilding because new_vdb=False but load failed."
                )

        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs: List[Document] = []

        normalized_raw_texts = self.hazmer(raw_texts)
        for text in normalized_raw_texts:
            docs.extend(Document(page_content=chunk) for chunk in splitter.split_text(text))

        self.vdb = FAISS.from_documents(
            documents=docs,
            embedding=self.embedding_model
            )
        self.vdb.save_local(vdb_dir)
        logging.getLogger(__name__).debug("Built new vector DB at %s", vdb_dir)

    def pdf_to_text(pdf_file):
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    @timer
    def load_vdb(self, query: str, vdb_dir: str = "faiss_vdb", k: int = 3) -> List[Document]:
        if not query:
            raise ValueError("query must be non-empty")
        self.vdb = FAISS.load_local(
                    folder_path=vdb_dir,
                    embeddings=self.embedding_model,
                    allow_dangerous_deserialization=True
                                    )       
        # self.retrieved_docs = self.vdb.similarity_search_with_relevance_scores(query, k=k)
        # for doc in self.retrieved_docs:
            # print(doc)
        self.retrieved_docs = self.vdb.similarity_search(query, k=k)

        return self.retrieved_docs

    @timer
    def extract_docs(self, retrieved_docs: Iterable[Document]) -> str:
        for doc in retrieved_docs:
            self.final_docs += doc.page_content + "\n\n"
        return self.final_docs
    