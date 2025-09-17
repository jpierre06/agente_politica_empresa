import os
from dotenv import load_dotenv
from typing import Optional

import re

def obter_api_key() -> str:
    # Carrega as variáveis do arquivo .env
    load_dotenv()

    # Lê a sua chave da variável de ambiente
    api_key = os.getenv("GOOGLE_API_KEY")

    return api_key

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()