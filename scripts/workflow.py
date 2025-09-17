from langgraph.graph import StateGraph, START, END

from scripts.agent import AgentState, AgentAction


# KEYWORDS_ABRIR_TICKET, triagem, perguntar_politica_RAG


class Workflow:

    def __init__(self, api_key: str):
        self.api_key = api_key  
        self.agent_action = AgentAction(self.api_key)


    def node_triagem(self, state: AgentState) -> AgentState:
        print("Executando nó de triagem...")
        # state isntância do tipo AgentState
        return {"triagem": self.agent_action.triagem(state["pergunta"])}

    
    def node_auto_resolver(self, state: AgentState) -> AgentState:
        print("Executando nó de auto_resolver...")
        resposta_rag = self.agent_action.perguntar_politica_RAG(state["pergunta"])

        update: AgentState = {
            "resposta": resposta_rag["answer"],
            "citacoes": resposta_rag.get("citacoes", []),
            "rag_sucesso": resposta_rag["contexto_encontrado"],
        }

        if resposta_rag["contexto_encontrado"]:
            update["acao_final"] = "AUTO_RESOLVER"

        return update

    
    def node_pedir_info(self, state: AgentState) -> AgentState:
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
    

    def node_abrir_chamado(self, state: AgentState) -> AgentState:
        print("Executando nó de abrir_chamado...")
        triagem = state["triagem"]

        return {
            "resposta": f"Abrindo chamado com urgência {triagem['urgencia']}. Descrição: {state['pergunta'][:140]}",
            "citacoes": [],
            "acao_final": "ABRIR_CHAMADO"
        }


    def decidir_pos_triagem(self, state: AgentState) -> str:
        print("Decidindo após a triagem...")
        decisao = state["triagem"]["decisao"]

        if decisao == "AUTO_RESOLVER": return "auto"
        if decisao == "PEDIR_INFO": return "info"
        if decisao == "ABRIR_CHAMADO": return "chamado"


    def decidir_pos_auto_resolver(self, state: AgentState) -> str:
        print("Decidindo após o auto_resolver...")

        if state.get("rag_sucesso"):
            print("Rag com sucesso, finalizando o fluxo.")
            return "ok"

        state_da_pergunta = (state["pergunta"] or "").lower()

        if any(k in state_da_pergunta for k in self.agent_action.KEYWORDS_ABRIR_TICKET):
            print("Rag falhou, mas foram encontradas keywords de abertura de ticket. Abrindo...")
            return "chamado"

        print("Rag falhou, sem keywords, vou pedir mais informações...")
        return "info"


    def criar_workflow(self):

        workflow = StateGraph(AgentState)

        workflow.add_node("triagem", self.node_triagem)
        workflow.add_node("auto_resolver", self.node_auto_resolver)
        workflow.add_node("pedir_info", self.node_pedir_info)
        workflow.add_node("abrir_chamado", self.node_abrir_chamado)

        workflow.add_edge(START, "triagem")
        workflow.add_conditional_edges("triagem", self.decidir_pos_triagem, {
            "auto": "auto_resolver",
            "info": "pedir_info",
            "chamado": "abrir_chamado"
        })

        workflow.add_conditional_edges("auto_resolver", self.decidir_pos_auto_resolver, {
            "info": "pedir_info",
            "chamado": "abrir_chamado",
            "ok": END
        })

        workflow.add_edge("pedir_info", END)
        workflow.add_edge("abrir_chamado", END)

        return workflow.compile()
