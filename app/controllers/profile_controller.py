from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.azure_llm_service import AzureLLMService
from app.services.pdf_extraction_service import PdfExtractionService

router = APIRouter(
    prefix="/profiles",
    tags=["Profiles"],
)

pdf_service = PdfExtractionService()
llm_service = AzureLLMService()


@router.post("/extract-cv")
async def extract_cv(file: UploadFile = File(...)):
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=422, detail="El archivo debe ser un PDF.")

    pdf_bytes = await file.read()
    extracted_text = pdf_service.extract_text_from_pdf(pdf_bytes)

    if not extracted_text:
        raise HTTPException(
            status_code=422,
            detail=(
                "No se pudo extraer texto del PDF. "
                "Si el archivo es escaneado, necesitas OCR antes de usar este endpoint."
            ),
        )

    candidate_profile = llm_service.extract_candidate_profile(extracted_text)

    if not candidate_profile:
        raise HTTPException(
            status_code=503,
            detail="Azure OpenAI no pudo estructurar el CV. Verifica la configuracion del modelo.",
        )

    return {
        "candidateProfile": candidate_profile.model_dump(by_alias=True, exclude_none=True),
        "rawTextLength": len(extracted_text),
        "usedAi": True,
    }