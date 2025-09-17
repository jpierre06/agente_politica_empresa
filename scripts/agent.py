
from pathlib import Path

from typing import TypedDict, Optional, List, Dict, Literal
from pydantic import BaseModel, Field
import re

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI

# Bibliotecas locais
from scripts.tools import clean_text


class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)


# Definir a estrutura do estado do Agente
# Tipo composto de dados
class AgentState(TypedDict, total = False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str


class AgentAction:

    def __init__(self, api_key: str):

        # Lê a API KEY identificada na aplicação
        self.GOOGLE_API_KEY = api_key

        ### Constantes e inicializações ###

        self.TRIAGEM_PROMPT = (
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

        self.KEYWORDS_ABRIR_TICKET = ["aprovação", "exceção", "liberação", "abrir ticket", "abrir chamado", "acesso especial"]
        
        self.path_politicas = './data/stage/unprocessed'
        print(f'Diretório com as políticas da empresa: {self.path_politicas}')


    def triagem(self, mensagem: str) -> Dict:

        llm_triagem = self.criar_llm_triagem()
        triagem_chain = llm_triagem.with_structured_output(TriagemOut)

        saida: TriagemOut = triagem_chain.invoke([
            SystemMessage(content=self.TRIAGEM_PROMPT),
            HumanMessage(content=mensagem)
        ])

        return saida.model_dump()

    # Formatadores

    def extrair_trecho(self, texto: str, query: str, janela: int = 240) -> str:
        """
        Extrai o trecho do documento de política relacionado com a pergunta.
        texto: Conteúdo da política consultado no banco de dados vetorial. Inclui metadados e dados em si
        query: Pergunta do usuário
        janela: Tamanho do trecho extraído
        return: Trecho extraído
        """
        txt = clean_text(texto)
        termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
        pos = -1
        for t in termos:
            pos = txt.lower().find(t)
            if pos != -1: break
        if pos == -1: pos = 0
        ini, fim = max(0, pos - janela//2), min(len(txt), pos + janela//2)
        return txt[ini:fim]


    def formatar_citacoes(self, docs_rel: List, query: str) -> List[Dict]:
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
            cites.append({"documento": src, "pagina": page, "trecho": self.extrair_trecho(d.page_content, query)})
        return cites[:3]


    def perguntar_politica_RAG(self, pergunta: str) -> Dict:
        # chunks recuperados do banco vetorial de acordo com a pergunda do usuário
        retriever = self.criar_retriever()

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
        document_chain = self.criar_document_chain()

        answer = document_chain.invoke({"input": pergunta,
                                        "context": docs_relacionados})

        txt = (answer or "").strip()

        if txt.rstrip(".!?") == "Não sei":
            return {"answer": "Não sei.",
                    "citacoes": [],
                    "contexto_encontrado": False}

        return {"answer": txt,
                "citacoes": self.formatar_citacoes(docs_relacionados, pergunta),
                "contexto_encontrado": True}


    def criar_llm_triagem(self):
        
        llm_triagem = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            api_key=self.GOOGLE_API_KEY
        )

        return llm_triagem


    def criar_chunks(self):
        # Carrega os documentos de políticas internas da empresa

        docs = []
        lista_arquivos = Path(self.path_politicas).glob("*.pdf")

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

        return chunks


    def criar_retriever(self):

        # TODO: Persistir o banco vetorial em disco
        # TODO: Consultar banco vetorial em disco

        # Criação do objeto com os parâmetros de criação dos vetores
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=self.GOOGLE_API_KEY
        )

        # Criação da base de dados vetorial com o embeddings e chunks
        chunks = self.criar_chunks()
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
        return retriever


    def criar_prompt_rag(self):
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
        return prompt_rag

    
    def criar_document_chain(self):
        # Uma cadeia de ações que o seu programa executa em sequência
        prompt_rag = self.criar_prompt_rag()
        llm_triagem = self.criar_llm_triagem()

        document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)
        
        return document_chain
