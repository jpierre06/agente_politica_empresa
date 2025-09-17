import streamlit as st

from scripts.workflow import Workflow 
from scripts.tools import obter_api_key


# --- INTERFACE WEB ---
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