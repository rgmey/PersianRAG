import os
import json
from openai import OpenAI
from jdatetime import datetime

from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import A4
import arabic_reshaper
from bidi.algorithm import get_display
from utils.decorators import timer
from reportlab.pdfgen import canvas
import textwrap

import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from hazm import Normalizer, sent_tokenize


class LLM:
    @timer
    def __init__(self) -> None:
        self.model_name = os.getenv('MODEL_NAME')
        self.base_url = os.getenv('BASE_URL')
        self.api_key = os.getenv('API_KEY')

    @timer
    def _log(self, prompt, output):
        datetime_now = datetime.now().strftime("%Y%m%d")
        log_path = '../logs'
        os.makedirs(log_path, exist_ok=True) 
        log_file = f"../logs/llm_calls{datetime_now}.log"
        log_dict = {
        "timestamp": datetime.now().strftime("%H%M%S"),
        "model": self.model_name,
        "input": prompt,
        "output": output,
        }   
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")
        pdf_path = '../pdfs'
        os.makedirs(pdf_path, exist_ok=True) 
        pdf_file = PDF(f'../pdfs/{datetime_now}_{datetime.now().strftime("%H%M%S")}.pdf')
        pdf_file.save_persian_pdf(self.model_name, prompt, output)

    @timer
    def _llm_init(self):
        if not self.api_key:
            raise RuntimeError("API_KEY not set")
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return client
    
    @timer
    def llm_call(self, prompt):
        client = self._llm_init()
        resp = client.chat.completions.create(
        model=self.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=8192,
        timeout=180
        )
        output = resp.choices[0].message.content
        self._log(prompt, output)
        print(output)
        return output



class PDF:
    """"Class for output PDF creation"""
    # Register Farsi font
    pdfmetrics.registerFont(TTFont("Vazirmatn", "Vazirmatn-Regular.ttf"))

    def __init__(self, file_name) -> None:
        self.file_name = file_name
        self.font_name = "Vazirmatn"
        self.font_size = 12
        self.page_width, self.page_height = A4
        self.right_margin = 40
        self.left_margin = 40
        self.top_margin = 50
        self.bottom_margin = 50
        self.line_spacing = 25
        self.max_chars_per_line = 90

    def save_persian_pdf(self, model_name, prompt: str, output: str):
        # Prepare full text with explicit \n you want to preserve
        prompt_text = 'Prompt:\n' + prompt.strip()
        output_text = f'Model Name:\n{model_name}\nModel Response:\n' + output.strip()
        full_text = prompt_text + "\n" + output_text

        # Split text by explicit newlines to preserve them as paragraph breaks
        lines_with_breaks = full_text.split('\n')

        c = canvas.Canvas(self.file_name, pagesize=A4)
        c.setFont(self.font_name, self.font_size)
        y = self.page_height - self.top_margin

        for line in lines_with_breaks:
            # Wrap each line separately, so \n acts as hard line break
            wrapped_lines = textwrap.wrap(line, width=self.max_chars_per_line) if line.strip() else ['']

            for wrapped_line in wrapped_lines:
                if y < self.bottom_margin:
                    c.showPage()
                    c.setFont(self.font_name, self.font_size)
                    y = self.page_height - self.top_margin

                if wrapped_line.strip() == '':
                    # Empty line, just add line spacing for paragraph break
                    y -= self.line_spacing
                    continue

                reshaped = arabic_reshaper.reshape(wrapped_line)
                bidi_line = get_display(reshaped)
                c.drawRightString(self.page_width - self.right_margin, y, bidi_line)
                y -= self.line_spacing

        c.save()
        print("PDF successfully saved to:", self.file_name)



class PDFtoFAISS:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", chunk_size=500):
        try:
            print(f"Loading embedding model: {model_name} ...")
            self.embedding_model = SentenceTransformer(model_name)
            print("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Could not load model '{model_name}': {e}")
        
        self.chunk_size = chunk_size
        self.normalizer = Normalizer()
        self.index = None
        self.id2text = {}

    def _read_pdf(self, file_path):
        """Extract text from PDF."""
        text = ""
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _preprocess_persian(self, text):
        """Normalize Persian text."""
        normalized_text = self.normalizer.normalize(text)
        sentences = sent_tokenize(normalized_text)
        return " ".join(sentences)

    def _split_text(self, text):
        """Split text into word chunks."""
        words = text.split()
        return [" ".join(words[i:i + self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

    @timer
    def process_pdf_to_faiss(self, pdf_path, index_path='faiss_index.index', normalize_persian=True):
        """
        Read PDF, preprocess, embed, and save to FAISS in one go.
        Stores text chunks inside FAISS index using IDs.
        """
        # Read PDF
        text = self._read_pdf(pdf_path)

        # Persian normalization if enabled
        if normalize_persian:
            text = self._preprocess_persian(text)

        # Split into chunks
        chunks = self._split_text(text)

        print(f"Number of chunks: {len(chunks)}")
        print(f"First chunk: {chunks[0] if chunks else 'No chunks found'}")

        # Encode
        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)

        # Create FAISS index with ID mapping
        dim = embeddings.shape[1]
        # index = faiss.IndexFlatL2(dim)
        # index = faiss.IndexIDMap(index)
        m = 32
        base_index = faiss.IndexHNSWFlat(dim, m)
        base_index.hnsw.efSearch = 40  # higher = better recall, slower
        base_index.hnsw.efConstruction = 80
        self.index = faiss.IndexIDMap(base_index)
        # Add vectors with IDs
        ids = np.arange(len(chunks))
        self.index.add_with_ids(embeddings, ids)

        # Store mapping
        self.id2text = {i: chunk for i, chunk in enumerate(chunks)}

        # Save index
        faiss.write_index(self.index, index_path)

        print(f"FAISS index saved to {index_path} with {len(chunks)} chunks.")

    @timer
    def load_index(self, index_path='faiss_index.index'):
        """Load FAISS index. In case we want to implement caching later."""
        self.index = faiss.read_index(index_path)

    @timer
    def search(self, query, top_k=5):
        """Search in FAISS index."""
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, ids = self.index.search(query_emb, top_k)
        results = [(self.id2text.get(int(i), "Not found"), float(dist)) for i, dist in zip(ids[0], distances[0])]
        return results
    
    def clear_data(self):
        """
        Clears the vector store and retriever to reset the system.
        """
        self.index = None
        self.model = None
