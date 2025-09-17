from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from scripts.tools import mover_arquivos


class PersistVetorialDB:

    def __init__(self, api_key: str):    
        self.GOOGLE_API_KEY = api_key
        
        self.path_politicas_unprocessed = './data/stage/unprocessed'
        self.path_politicas_processed = './data/stage/processed'
        self.path_politicas_error_processed = './data/stage/error_processed'

        self.faiss_dir = './data/faiss_store'
        Path(self.faiss_dir).mkdir(parents=True, exist_ok=True)


    def criar_db(self):
        lista_arquivos = list(Path(self.path_politicas_unprocessed).glob("*.pdf"))
        
        if len(lista_arquivos) > 0:
            try:
                chunks = self.criar_chunks(lista_arquivos)
                self.salvar_chunks_faiss(chunks)
                # Sucesso: mover para processed
                mover_arquivos(lista_arquivos, self.path_politicas_processed, erro=False)
            except Exception as e:
                print(f'Erro ao criar base vetorial: {e}')
                # Falha: mover para error_processed com timestamp
                
                # TODO: Descomentar a linha abaixo
                # mover_arquivos(lista_arquivos, self.path_politicas_error_processed, erro=True)


    def criar_chunks(self, lista_arquivos):
        # Carrega os documentos de políticas internas da empresa

        docs = []
        
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


    def salvar_chunks_faiss(self, chunks):
        """
        Salva os chunks recebidos em uma base FAISS local.
        """
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=self.GOOGLE_API_KEY
        )
        faiss_path = self.faiss_dir
        #faiss_path.mkdir(parents=True, exist_ok=True)
        try:
            print('Criando index FAISS a partir dos documentos...')
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(str(faiss_path))
            print(f'Índice FAISS salvo em: {faiss_path}')
        except Exception as e:
            print(f'Erro ao salvar índice FAISS: {e}')


    def consultar_faiss_por_texto(self, texto, search_type="similarity_score_threshold", k=4, score_threshold=0.3):
        """
        Consulta a base FAISS local pelo texto fornecido.
        Retorna os documentos mais similares.
        """
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=self.GOOGLE_API_KEY
        )
        
        faiss_path = Path(self.faiss_dir)
        try:
            if not any(faiss_path.iterdir()):
                print('Base FAISS não encontrada.')
                return []
            vectorstore = FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs={"score_threshold": score_threshold, "k": k}
            )
            resultados = retriever.invoke(texto)
            return resultados
        except Exception as e:
            print(f'Erro ao consultar FAISS: {e}')
            return []
