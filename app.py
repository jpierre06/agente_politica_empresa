import os
from pathlib import Path
from dotenv import load_dotenv

from typing import TypedDict, Optional, List, Dict, Literal
from pydantic import BaseModel, Field
import re

import streamlit as st

from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI



TRIAGEM_PROMPT = (
    "Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?").\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral").\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado (Ex: "Quero exceção para trabalhar 5 dias remoto.", "Solicito liberação para anexos externos.", "Por favor, abra um chamado para o RH.").'
    "Analise a mensagem e decida a ação mais apropriada."
)

class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)


def obter_api_key():
    # Carrega as variáveis do arquivo .env
    load_dotenv()

    # Lê a sua chave da variável de ambiente
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        with st.sidebar:  
            api_key = st.text_input("Gemini API Key", key="chatbot_api_key", type="password")
            "[Get an Gemini API key](https://aistudio.google.com/app/apikey)"
            
    return api_key

# Lê a sua chave da variável de ambiente
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

GOOGLE_API_KEY = obter_api_key()


llm_triagem = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    api_key=GOOGLE_API_KEY
)

triagem_chain = llm_triagem.with_structured_output(TriagemOut)

def triagem(mensagem: str) -> Dict:
    saida: TriagemOut = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT),
        HumanMessage(content=mensagem)
    ])

    return saida.model_dump()


path_politicas = './data/stage/unprocessed'
print(f'Diretório com as políticas da empresa: {path_politicas}')


docs = []
lista_arquivos = Path(path_politicas).glob("*.pdf")

# Converte cada arquivo pdf encontrado na lista de arquivos para TXT
for n in lista_arquivos:
    try:
        loader = PyMuPDFLoader(str(n))
        docs.extend(loader.load())
        print(f"Carregado com sucesso arquivo {n.name}")

    except Exception as e:
        print(f"Erro ao carregar arquivo {n.name}: {e}")

print(f"Total de documentos carregados: {len(docs)}")


# Overlap permite que os últimos 30 caracteres do chuck anterior faça parte do
# próximo chunk para evitar que uma ideia se perca em um chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

chunks = splitter.split_documents(docs)
print(f'Foram criados {len(chunks)} chunks')


# Criação do objeto com os parâmetros de criação dos vetores
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)


# Criação da base de dados vetorial com o embeddings e chunks
vectorstore = FAISS.from_documents(chunks, embeddings)

# Defini o grau de similaridade entre o termo pesquisado e o chunks armazendos
# no banco vetorial do FAISS. Retorna apenas as 4 primeiras correspondências.
# "score_threshold" seria um índice de divergência,
# ou seja divergência de 0.3 ou similiridade de 0.7
# o conceito inverso? similiridade de 0.3 e divergência de 0.7
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold":0.3, "k": 4}
)


llm_triagem = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    api_key=GOOGLE_API_KEY
)

# O {input} é a pergunta do usuário em si
# O {context} é o espaço reservado para os pedaços de texto (chunks) mais relevantes
# que o nosso "pesquisador" (retriever) buscou no banco de dados FAISS.

prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
     "Responda SOMENTE com base no contexto fornecido. "
     "Se não houver base suficiente, responda apenas 'Não sei'."),

    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])


# Uma cadeia de ações que o seu programa executa em sequência
document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)


# Formatadores
def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    """
      Extrai o trecho do documento de política relacionado com a pergunta.
      texto: Conteúdo da política consultado no banco de dados vetorial. Inclui metadados e dados em si
      query: Pergunta do usuário
      janela: Tamanho do trecho extraído
      return: Trecho extraído
    """
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1: break
    if pos == -1: pos = 0
    ini, fim = max(0, pos - janela//2), min(len(txt), pos + janela//2)
    return txt[ini:fim]

def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    """
      Extrai metadados de registro(s) de uma base de dados vetorial.
      docs_rel: Documentos encontrados no banco de dados vetorial
      query: Pergunta do usuário
      return: Lista de citações formatadas
    """
    cites, seen = [], set()
    for d in docs_rel:
        src = Path(d.metadata.get("source","")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({"documento": src, "pagina": page, "trecho": extrair_trecho(d.page_content, query)})
    return cites[:3]

def perguntar_politica_RAG(pergunta: str) -> Dict:
    # chunks recuperados do banco vetorial de acordo com a pergunda do usuário
    docs_relacionados = retriever.invoke(pergunta)

    # Se na base de dados não houver nenhum documento que referencie a pergunta
    # é retornado não sei
    if not docs_relacionados:
        return {"answer": "Não sei.",
                "citacoes": [],
                "contexto_encontrado": False}

    # É criado um contexto com a pergunta do usuário e os documentos
    # relacionados e enviado para a cadeia de ações do agente
    # "input" e "context" são usados no prompt_rag que faz parte do
    # document_chain
    answer = document_chain.invoke({"input": pergunta,
                                    "context": docs_relacionados})

    txt = (answer or "").strip()

    if txt.rstrip(".!?") == "Não sei":
        return {"answer": "Não sei.",
                "citacoes": [],
                "contexto_encontrado": False}

    return {"answer": txt,
            "citacoes": formatar_citacoes(docs_relacionados, pergunta),
            "contexto_encontrado": True}


# Definir a estrutura do estado do Agente
# Tipo composto de dados
class AgentState(TypedDict, total = False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str


def node_triagem(state: AgentState) -> AgentState:
    print("Executando nó de triagem...")
    # state isntância do tipo AgentState
    return {"triagem": triagem(state["pergunta"])}

def node_auto_resolver(state: AgentState) -> AgentState:
    print("Executando nó de auto_resolver...")
    resposta_rag = perguntar_politica_RAG(state["pergunta"])

    update: AgentState = {
        "resposta": resposta_rag["answer"],
        "citacoes": resposta_rag.get("citacoes", []),
        "rag_sucesso": resposta_rag["contexto_encontrado"],
    }

    if resposta_rag["contexto_encontrado"]:
        update["acao_final"] = "AUTO_RESOLVER"

    return update

def node_pedir_info(state: AgentState) -> AgentState:
    print("Executando nó de pedir_info...")
    faltantes = state["triagem"].get("campos_faltantes", [])
    if faltantes:
        detalhe = ",".join(faltantes)
    else:
        detalhe = "Tema e contexto específico"

    return {
        "resposta": f"Para avançar, preciso que detalhe: {detalhe}",
        "citacoes": [],
        "acao_final": "PEDIR_INFO"
    }

def node_abrir_chamado(state: AgentState) -> AgentState:
    print("Executando nó de abrir_chamado...")
    triagem = state["triagem"]

    return {
        "resposta": f"Abrindo chamado com urgência {triagem['urgencia']}. Descrição: {state['pergunta'][:140]}",
        "citacoes": [],
        "acao_final": "ABRIR_CHAMADO"
    }

KEYWORDS_ABRIR_TICKET = ["aprovação", "exceção", "liberação", "abrir ticket", "abrir chamado", "acesso especial"]

def decidir_pos_triagem(state: AgentState) -> str:
    print("Decidindo após a triagem...")
    decisao = state["triagem"]["decisao"]

    if decisao == "AUTO_RESOLVER": return "auto"
    if decisao == "PEDIR_INFO": return "info"
    if decisao == "ABRIR_CHAMADO": return "chamado"

def decidir_pos_auto_resolver(state: AgentState) -> str:
    print("Decidindo após o auto_resolver...")

    if state.get("rag_sucesso"):
        print("Rag com sucesso, finalizando o fluxo.")
        return "ok"

    state_da_pergunta = (state["pergunta"] or "").lower()

    if any(k in state_da_pergunta for k in KEYWORDS_ABRIR_TICKET):
        print("Rag falhou, mas foram encontradas keywords de abertura de ticket. Abrindo...")
        return "chamado"

    print("Rag falhou, sem keywords, vou pedir mais informações...")
    return "info"


def criar_workflow(agent_state):

    workflow = StateGraph(agent_state)

    workflow.add_node("triagem", node_triagem)
    workflow.add_node("auto_resolver", node_auto_resolver)
    workflow.add_node("pedir_info", node_pedir_info)
    workflow.add_node("abrir_chamado", node_abrir_chamado)

    workflow.add_edge(START, "triagem")
    workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
        "auto": "auto_resolver",
        "info": "pedir_info",
        "chamado": "abrir_chamado"
    })

    workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
        "info": "pedir_info",
        "chamado": "abrir_chamado",
        "ok": END
    })

    workflow.add_edge("pedir_info", END)
    workflow.add_edge("abrir_chamado", END)

    return workflow.compile()

grafo = criar_workflow(AgentState)



# --- INTERFACE WEB ---
st.title("Agente de IA de Políticas Internas")

pergunta_usuario = st.text_input("Digite sua dúvida sobre políticas internas:")

# Cria duas colunas para posicionar os botões lado a lado
col1, col2 = st.columns(2)

# Adiciona o botão "Obter Resposta" na primeira coluna
with col1:
    if st.button("Obter Resposta") and pergunta_usuario:
        with st.spinner("Analisando sua pergunta..."):
            resposta_final = grafo.invoke({"pergunta": pergunta_usuario})

        # Exibe a resposta e as citações
        st.subheader("Resposta")
        st.write(resposta_final.get("resposta"))

        if resposta_final.get("citacoes"):
            st.subheader("Citações")
            for citacao in resposta_final.get("citacoes"):
                st.markdown(
                    f"""
                    - **Documento:** {citacao['documento']}
                    - **Página:** {citacao['pagina']}
                    - **Trecho:** {citacao['trecho']}
                    """
                )

# Adiciona o botão "Limpar" na segunda coluna
with col2:
    if st.button("Limpar"):
        st.subheader("")    