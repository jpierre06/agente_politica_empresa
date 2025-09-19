
import logging
import io
import streamlit as st
from scripts.workflow import Workflow 
from scripts.tools import obter_api_key



# --- LOGGING PARA STREAMLIT ---
log_buffer = io.StringIO()
handler = logging.StreamHandler(log_buffer)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
root_logger = logging.getLogger()
if not any(isinstance(h, logging.StreamHandler) and h.stream == log_buffer for h in root_logger.handlers):
    root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO)

st.title("Agente de IA de Políticas Internas")

# Obter API Key
api_key = obter_api_key()

if not api_key:
    with st.sidebar:  
        api_key_app = st.text_input("Gemini API Key", key="chatbot_api_key", type="password")
        "[Get an Gemini API key](https://aistudio.google.com/app/apikey)"

    api_key = api_key_app
            

# Inicializa o workflow
workflow = Workflow(api_key)
grafo = workflow.criar_workflow()


pergunta_usuario = st.text_input("Digite sua dúvida sobre políticas internas:")

# Cria duas colunas para posicionar os botões lado a lado
col1, col2 = st.columns(2)

# Adiciona o botão "Obter Resposta" na primeira coluna
with col1:
    if st.button("Obter Resposta") and pergunta_usuario:
        with st.spinner("Analisando sua pergunta..."):
            resposta_final = grafo.invoke({"pergunta": pergunta_usuario})
            triagem = resposta_final.get("triagem", {})
            triagem_decisao = (f"DECISÃO: {triagem.get('decisao')}")
            triagem_urgencia = (f"URGÊNCIA: {triagem.get('urgencia')}")
            triagem_acao_final = (f"AÇÃO FINAL: {resposta_final.get('acao_final')}")

        # Exibe a resposta e as citações
        st.subheader("Decisão da Triagem")
        st.write(triagem_decisao)
        st.write(triagem_urgencia)
        st.write(triagem_acao_final)

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

# Caixa de seleção para exibir logs
exibir_logs = st.checkbox("Exibir logs da aplicação", value=False)
if exibir_logs:
    st.subheader("Logs da aplicação")
    st.text_area(
        label="Inicio dos Logs", 
        value=log_buffer.getvalue(), 
        height=100, 
        max_chars=None, 
        key="log_text_area",
        disabled=True
    )
