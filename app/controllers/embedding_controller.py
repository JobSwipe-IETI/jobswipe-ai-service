from fastapi import APIRouter
from app.models.schemas import TextRequest
from app.services.embedding_service import EmbeddingService
from app.models.schemas import MatchRequest
from app.services.matching_service import MatchingService
from app.services.azure_llm_service import AzureLLMService

router = APIRouter(
    prefix="/embeddings",
    tags=["Embeddings"]
)

service = EmbeddingService()
matching_service = MatchingService()


@router.post("/")
def generate_embedding(request: TextRequest):
    result = service.generate_embedding(request.text)
    return {"result": result}

@router.post("/match")
def match_texts(request: MatchRequest):

    emb1 = service.generate_embedding(request.text1)
    emb2 = service.generate_embedding(request.text2)
    score = matching_service.calculate_similarity(emb1, emb2)

    return {
        "similarity_score": score
    }

