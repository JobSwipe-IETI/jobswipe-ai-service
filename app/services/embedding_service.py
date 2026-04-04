import os

from openai import AzureOpenAI


class EmbeddingService:
    """
    Generate embeddings using Azure OpenAI (text-embedding-ada-002).
    """

    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

        self._client = None
        self._init_error = None
        if self.is_configured():
            try:
                self._client = AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.endpoint,
                    api_version=self.api_version,
                )
            except Exception as e:
                self._init_error = str(e)
                self._client = None

    def is_configured(self) -> bool:
        return bool(self.api_key and self.endpoint and self.deployment)

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for text using Azure OpenAI.
        """
        if not self._client:
            raise RuntimeError(
                "Azure OpenAI not configured. "
                "Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT."
            )

        # Truncate text to avoid token limit issues.
        max_chars = 30000
        text = text[:max_chars]

        try:
            response = self._client.embeddings.create(
                model=self.deployment,
                input=text,
            )

            if not response.data:
                raise RuntimeError("Azure OpenAI returned empty embedding response.")

            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding with Azure: {str(e)}")