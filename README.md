# 🎓 O projeto faz parte da Imersão Dev Agentes de IA com Gemini

O projeto é uma evolução da atividade realizada na Imersão

<br>

# 🤖 Agente de IA de Políticas Internas

![Python](https://img.shields.io/badge/Python-3.11.7-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-ffffff?logo=langchain&logoColor=green)
![LangFlow](https://img.shields.io/badge/LangFlow-ffffff?logo=langflow&logoColor=blue)
![Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=flat&logo=google&logoColor=white)
![](https://img.shields.io/badge/FAISS-blue?logo=facebook&style=flat)

Este projeto é um assistente IA para consulta e automação de respostas sobre políticas internas de uma empresa, utilizando IA generativa, busca semântica e interface web com Streamlit.

Os arquivos de Políticas Internas utilizado no projeto são fictícios e não devem ser interpretados como algum tipo de regra trabalhista.

<br>

## 🎯 Funcionalidades
- Consulta automática de políticas internas em PDF
- Busca semântica com FAISS e embeddings Gemini
- Triagem automática de perguntas (auto-resposta, pedir mais informações ou abrir chamado)
- Interface web interativa (Streamlit)
- Log detalhado das operações, com opção de visualização na interface
- Movimentação automática dos arquivos processados

<br>

## 🗂️ Estrutura do Projeto
```
├── app.py                      # Interface principal Streamlit
├── scripts/
│   ├── agent.py                # Lógica do agente e integração RAG
│   ├── persist_vetorial_db.py  # Persistência e consulta FAISS
│   ├── tools.py                # Utilitários (API key, mover arquivos, etc)
│   └── workflow.py             # Orquestração do fluxo de decisão
├── data/
│   └── stage/
│       ├── unprocessed/        # PDFs aguardando processamento
│       ├── processed/          # PDFs processados com sucesso
│       └── error_processed/    # PDFs com erro no processamento
│   └── faiss_store/            # Base vetorial FAISS
├── requirements.txt            # Dependências do projeto
```

<br>

## ▶️ Como executar
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Defina sua chave da API Gemini no arquivo `.env` ou via variável de ambiente `GOOGLE_API_KEY`.
3. Coloque os arquivos PDF de políticas em `data/stage/unprocessed/`.
4. Execute a aplicação:
   ```bash
   python -m streamlit run app.py
   ```
5. Acesse a interface web em `http://localhost:8501`.

6. Porta alternativa
   * Por padrão o Streamlit executa na porta 8501. Caso queira executar a aplicação em uma porta alternativa execute:

   ```bash
   python -m streamlit run app.py --server.port=8502 --server.address=0.0.0.0
   ```

<br>

## 📌 Observações
- Os logs podem ser visualizados na interface, ativando a opção "Exibir logs da aplicação".
- Os arquivos PDF são movidos automaticamente para as pastas `processed` ou `error_processed` após o processamento.
- O índice FAISS é persistido em disco para acelerar consultas futuras.

<br>

## ⚙️ Requisitos
- Python 3.11+
- Dependências do `requirements.txt`

### Bibliotecas Principais
- `streamlit` – interface web
- `langchain` – framework de de LLMs
- `langgraph` – framework de fluxos de agentes de IA
- `faiss-cpu` – mecanismo de banco de dados vetorial
- `google-generativeai` – api de IA Gen da Google

<br>

## ☁️ Publicação `Streamlit.io`

Para publicação da aplicação no serviço do `https://streamlit.io/` é necessário ter a base de dados com as políticas da empresa.

Atualmente o `.gitignore` está configurado para não gerenciar o conteúdo do diretório data e consequentemente não será publicado nenhuma base de dados com as políticas internas.

* ### Caso queria publicar as políticas no formato .PDF para posterior processamento, remover ou comentar a linha abaixo no `.gitignore`

`data/stage/unprocessed/`

* ### Caso queria publicar as políticas em banco de dados de vetor do FAISS, remover ou comentar a linha abaixo no `.gitignore`

`data/faiss_store/`

<br>


## 📬 Contato

Se você tiver dúvidas, sugestões ou quiser contribuir com melhorias, sinta-se à vontade para entrar em contato ou abrir uma issue no repositório.

### 🛠️ Adaptado e desenvolvido por: Jean Pierre
### 📧 jps.data.analise@gmail.com
### 💼 https://www.linkedin.com/in/jeanpierresantana
### 🧩 https://github.com/jpierre06
