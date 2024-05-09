import os

try:
    from langchain_openai import ChatOpenAI
    openai_backend_available = True
except ImportError:
    openai_backend_available = False

try:
    from langchain_groq import ChatGroq
    groq_backend_available = True
except ImportError:
    groq_backend_available = False


backend_lst = [openai_backend_available, groq_backend_available]


if not any(backend_lst):
    raise ImportError(
        "None of the supported backends is installed. Install one of the following: langchain_openai, langchain_groq"
    )


def get_backend(openapi_token=None, groq_token=None, **kwargs):
    openapi_env_token = os.getenv("OPENAI_API_KEY", None)
    if openapi_token is None and openapi_env_token is not None:
        openapi_token = openapi_env_token
    groq_env_token = os.getenv("GROQ_API_KEY", None)
    if groq_token is None and groq_env_token is not None:
        groq_token = groq_env_token
    if not any([openapi_token, groq_token]):
        raise ValueError("The API token has to be provided either as input parameter or as environment variable.")

    if groq_backend_available and groq_token:
        if len(kwargs) == 0:
            kwargs = {"model_name": "llama3-8b-8192", "temperature": 0}
        return ChatGroq(groq_api_key=groq_token, **kwargs)
    elif openai_backend_available and openapi_token:
        if len(kwargs) == 0:
            kwargs = {"model": "gpt-3.5-turbo", "temperature": 0}
        return ChatOpenAI(openai_api_key=openapi_token, **kwargs)
    elif groq_token:
        raise ImportError("Groq token is provided but langchain_groq is not installed.")
    elif openapi_token:
        raise ImportError("OpenAI token is provided but langchain_openai is not installed.")
