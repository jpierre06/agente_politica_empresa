import streamlit as st

from scripts.RAG import triagem, resposta, obter_api_key
from scripts.workflow import criar_workflow, AgentState 

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