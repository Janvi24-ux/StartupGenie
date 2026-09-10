"""
IBM Granite Integration – LLM Client
Wraps ibm-watsonx-ai to call IBM Granite models with retry logic.
"""

import logging
from typing import Dict, Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)

# Placeholder strings that indicate the user has not set real credentials
_PLACEHOLDER_PATTERNS = (
    "your_ibm",
    "your_watsonx",
    "placeholder",
    "api_key_here",
    "project_id_here",
    "<your",
)


def _looks_like_placeholder(value: str) -> bool:
    """Return True if value is clearly a template placeholder, not a real credential."""
    v = value.strip().lower()
    if not v:
        return True
    return any(pat in v for pat in _PLACEHOLDER_PATTERNS)


def _friendly_auth_error(original: Exception) -> RuntimeError:
    """Convert a raw IBM SDK auth exception into an actionable message."""
    msg = str(original)
    if "400" in msg or "IAM Token" in msg or "iam" in msg.lower():
        return RuntimeError(
            "IBM watsonx authentication failed (HTTP 400 from IAM). "
            "Your API key is invalid or has been revoked. "
            "Steps to fix:\n"
            "  1. Open your .env file.\n"
            "  2. Set WATSONX_API_KEY to a real IBM Cloud API key "
            "(create one at https://cloud.ibm.com/iam/apikeys).\n"
            "  3. Set WATSONX_PROJECT_ID to your watsonx.ai project ID "
            "(found under your project → Manage → General).\n"
            "  4. Restart the server."
        )
    if "403" in msg or "Forbidden" in msg:
        return RuntimeError(
            "IBM watsonx authorisation denied (HTTP 403). "
            "Your API key does not have access to the specified project or model. "
            "Check that WATSONX_PROJECT_ID is correct and that the IBM Granite "
            "model is enabled for your account."
        )
    if "404" in msg or "model" in msg.lower():
        return RuntimeError(
            f"IBM watsonx model not found. "
            f"Check that GRANITE_MODEL_ID is a valid model ID for your region. "
            f"Original error: {msg}"
        )
    return RuntimeError(f"IBM watsonx error: {msg}")


class GraniteClient:
    """
    Client for IBM Granite LLM via watsonx.ai.

    Credential validation happens eagerly on the first call so the server
    starts successfully even without valid keys, but returns a clear error
    message to the user at generation time.
    """

    def __init__(
        self,
        api_key: str,
        project_id: str,
        url: str = "https://us-south.ml.cloud.ibm.com",
        model_id: str = "ibm/granite-13b-instruct-v2",
    ):
        self.api_key = api_key
        self.project_id = project_id
        self.url = url
        self.model_id = model_id
        self._model = None

    def _uses_chat_template(self) -> bool:
        """
        Granite 3.x and 4.x models use a special chat template format.
        Granite 13b/20b/34b use plain-text prompts.
        """
        mid = self.model_id.lower()
        # granite-4-*, granite-3-* use the chat template
        return (
            "granite-4" in mid
            or "granite-3" in mid
            or "granite-7b-lab" in mid
        )

    def wrap_prompt(self, user_prompt: str) -> str:
        """
        Wrap the prompt in the correct format for the model family.

        Granite 4/3 expect:
            <|system|>\\n{system}\\n<|user|>\\n{user}\\n<|assistant|>\\n

        Granite 13b and older accept a plain prompt.
        """
        if self._uses_chat_template():
            system = (
                "You are StartupGenie, an expert AI startup advisor. "
                "Generate detailed, structured, and actionable startup blueprints. "
                "Label every claim as [RETRIEVED FACT], [ESTIMATE], [ASSUMPTION], or [RECOMMENDATION]."
            )
            return f"<|system|>\n{system}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
        return user_prompt

    def _validate_credentials(self) -> None:
        """
        Raise a clear RuntimeError immediately if credentials look like
        placeholders, before making any network call.
        """
        if _looks_like_placeholder(self.api_key):
            raise RuntimeError(
                "WATSONX_API_KEY is not set. "
                "Open your .env file and replace the placeholder with a real "
                "IBM Cloud API key (https://cloud.ibm.com/iam/apikeys)."
            )
        if _looks_like_placeholder(self.project_id):
            raise RuntimeError(
                "WATSONX_PROJECT_ID is not set. "
                "Open your .env file and replace the placeholder with your "
                "watsonx.ai project ID (project → Manage → General)."
            )

    def _get_model(self):
        """
        Lazily initialise the watsonx ModelInference instance.
        The IBM SDK fetches an IAM token during APIClient construction,
        so authentication errors surface here on the first call.
        """
        if self._model is not None:
            return self._model

        # Fast-fail with a human-readable message before any HTTP calls.
        self._validate_credentials()

        try:
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

            credentials = Credentials(
                api_key=self.api_key,
                url=self.url,
            )

            params = {
                GenParams.MAX_NEW_TOKENS: 4096,
                GenParams.MIN_NEW_TOKENS: 50,
                GenParams.TEMPERATURE: 0.7,
                GenParams.TOP_P: 0.9,
                GenParams.REPETITION_PENALTY: 1.05,
            }

            self._model = ModelInference(
                model_id=self.model_id,
                credentials=credentials,
                project_id=self.project_id,
                params=params,
                validate=False,   # skip model-ID ping; auth still happens via APIClient
            )
            logger.info("IBM Granite model initialised: %s", self.model_id)

        except ImportError as e:
            raise RuntimeError(
                "ibm-watsonx-ai is not installed. Run: pip install ibm-watsonx-ai"
            ) from e
        except RuntimeError:
            # Re-raise our own friendly errors unchanged.
            raise
        except Exception as e:
            raise _friendly_auth_error(e) from e

        return self._model

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        # Only retry on transient network errors, NOT on auth failures.
        retry=retry_if_exception_type(ConnectionError),
        reraise=True,
    )
    def generate(self, prompt: str) -> str:
        """
        Send a prompt to IBM Granite and return the generated text.

        Args:
            prompt: The complete prompt string.

        Returns:
            Generated text string from the model.

        Raises:
            RuntimeError: On auth failure or after retries.
        """
        model = self._get_model()
        logger.info("Sending prompt to IBM Granite (model=%s)...", self.model_id)

        # Apply the correct prompt format for the model family (chat vs plain)
        formatted_prompt = self.wrap_prompt(prompt)

        try:
            response = model.generate_text(prompt=formatted_prompt)
            if isinstance(response, str):
                return response.strip()
            if isinstance(response, dict):
                results = response.get("results", [{}])
                if results:
                    return results[0].get("generated_text", "").strip()
            return str(response).strip()
        except RuntimeError:
            raise
        except Exception as e:
            logger.error("IBM Granite generation error: %s", e)
            raise _friendly_auth_error(e) from e

    def health_check(self) -> Dict[str, Any]:
        """
        Return the health status without making a live LLM call.
        Reports whether credentials are configured (not necessarily valid).
        """
        try:
            self._validate_credentials()
            return {
                "status": "configured",
                "model_id": self.model_id,
                "detail": "Credentials are set. Connectivity verified on first generate() call.",
            }
        except RuntimeError as e:
            return {"status": "error", "model_id": self.model_id, "detail": str(e)}
