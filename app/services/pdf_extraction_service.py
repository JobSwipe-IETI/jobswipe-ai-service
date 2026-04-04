from io import BytesIO

from pypdf import PdfReader


class PdfExtractionService:
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        reader = PdfReader(BytesIO(pdf_bytes))
        pages_text = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages_text.append(page_text.strip())

        return "\n\n".join(pages_text).strip()