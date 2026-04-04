import app.services.pdf_extraction_service as pdf_module
from app.services.pdf_extraction_service import PdfExtractionService


class _FakePage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakePdfReader:
    def __init__(self, stream):
        self.pages = [_FakePage("First page"), _FakePage("Second page")]


def test_extract_text_from_pdf_joins_pages(monkeypatch):
    monkeypatch.setattr(pdf_module, "PdfReader", _FakePdfReader)

    service = PdfExtractionService()
    result = service.extract_text_from_pdf(b"%PDF-1.4 fake")

    assert result == "First page\n\nSecond page"


def test_extract_text_from_pdf_ignores_empty_pages(monkeypatch):
    class _EmptyPdfReader:
        def __init__(self, stream):
            self.pages = [_FakePage(""), _FakePage("   ")]

    monkeypatch.setattr(pdf_module, "PdfReader", _EmptyPdfReader)

    service = PdfExtractionService()
    result = service.extract_text_from_pdf(b"%PDF-1.4 fake")

    assert result == ""