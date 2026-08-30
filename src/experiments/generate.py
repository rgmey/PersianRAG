import os
import json
from openai import OpenAI
from jdatetime import datetime

# pdf imports
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import A4
import arabic_reshaper
from bidi.algorithm import get_display
from utils.decorators import timer
from reportlab.pdfgen import canvas
import textwrap


class LLM:
    def __init__(self) -> None:
        self.model_name = os.getenv('MODEL_NAME')
        self.base_url = os.getenv('BASE_URL')
        self.api_key = os.getenv('API_KEY')

    def _log(self, prompt, output):
        datetime_now = datetime.now().strftime("%Y%m%d")
        log_file = f"../logs/llm_calls{datetime_now}.log"
        log_dict = {
        "timestamp": datetime.now().strftime("%H%M%S"),
        "model": self.model_name,
        "input": prompt,
        "output": output,
        }   
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")
        pdf_file = PDF(f'../pdfs/{datetime_now}_{datetime.now().strftime("%H%M%S")}.pdf')
        pdf_file.save_persian_pdf(self.model_name, prompt, output)

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
        print("✅ PDF successfully saved to:", self.file_name)


