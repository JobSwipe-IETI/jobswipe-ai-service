from fastapi import APIRouter
from fastapi import HTTPException
from app.models.schemas import MatchRequest
from app.models.schemas import MatchResponse
from app.models.schemas import TextRequest
from app.services.embedding_service import EmbeddingService
from app.services.azure_llm_service import AzureLLMService
from app.services.matching_service import MatchingService

router = APIRouter(
    prefix="/embeddings",
    tags=["Embeddings"]
)

service = EmbeddingService()
matching_service = MatchingService()
llm_service = AzureLLMService()


@router.post("/")
def generate_embedding(request: TextRequest):
    result = service.generate_embedding(request.text)
    return {"result": result}


@router.post("/match")
def match_texts(request: MatchRequest) -> MatchResponse:
    candidate_text = request.candidate_text or request.text1
    vacancy_text = request.vacancy_text or request.text2

    if not candidate_text and request.candidate_profile:
        candidate_text = matching_service.build_candidate_text(
            request.candidate_profile.model_dump()
        )

    if not vacancy_text and request.vacancy_profile:
        vacancy_text = matching_service.build_vacancy_text(
            request.vacancy_profile.model_dump()
        )

    if not candidate_text or not vacancy_text:
        raise HTTPException(
            status_code=422,
            detail=(
                "Debe enviar candidate_text y vacancy_text, "
                "candidate_profile y vacancy_profile, "
                "o text1 y text2 para compatibilidad."
            ),
        )

    candidate_embedding = service.generate_embedding(candidate_text)
    vacancy_embedding = service.generate_embedding(vacancy_text)
    similarity_score = matching_service.calculate_similarity(
        candidate_embedding,
        vacancy_embedding,
    )

    compatibility_percentage = matching_service.to_compatibility_percentage(
        similarity_score
    )
    compatibility_level = matching_service.compatibility_level(compatibility_percentage)

    llm_feedback = llm_service.generate_match_feedback(
        candidate_text=candidate_text,
        vacancy_text=vacancy_text,
        compatibility_percentage=compatibility_percentage,
    )
    used_llm_feedback = bool(llm_feedback)
    feedback = llm_feedback or matching_service.generate_rule_based_feedback(
        compatibility_percentage
    )

    return MatchResponse(
        similarity_score=round(similarity_score, 4),
        compatibility_percentage=compatibility_percentage,
        compatibility_level=compatibility_level,
        feedback=feedback,
        used_llm_feedback=used_llm_feedback,
    )

