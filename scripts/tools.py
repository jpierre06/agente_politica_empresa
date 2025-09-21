import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

import re
from datetime import datetime

def obter_api_key(value_env = "GOOGLE_API_KEY") -> str:
    # Carrega as variáveis do arquivo .env
    load_dotenv()

    try:
        # Lê a sua chave da variável de ambiente
        api_key = os.getenv(value_env)
        return api_key
    except Exception as e:
        return None


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def mover_arquivos(arquivos, destino, erro=False):
    """Move arquivos para o destino. Se erro=True, adiciona timestamp ao nome."""
    Path(destino).mkdir(parents=True, exist_ok=True)
    for arquivo in arquivos:
        nome = Path(arquivo).name
        if erro:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            nome = f"{timestamp}_{nome}"
        destino_final = Path(destino) / nome
        try:
            Path(arquivo).rename(destino_final)
            print(f"Movido: {arquivo} -> {destino_final}")
        except Exception as e:
            print(f"Erro ao mover {arquivo}: {e}")