from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
from crewai_tools import SerperDevTool
import time

load_dotenv()

ferramenta_pesquisa = SerperDevTool()

modelo = LLM (
    model="gemini/gemini-3.8-flash"
)

pesquisador = Agent (
    role="Pesquisador de dados e Informações sobre política", 
    goal="Encontrar dados, informações e pesquisa confiáveis sobre política",
    backstory="Você é um analista político especializado em pesquisar e organizar e informações de forma objetiva e baseada em fontes.",
    verbose=True,
    llm=modelo,
    tools=[ferramenta_pesquisa]  
)

tarefa_pesquisa = Task(
    description=(
        "Pesquise dados de pesquisas eleitorais registradas sobre a eleição presidencial brasileira. "
        "Identifique a intenção de voto para cada candidato e percentual de votos brancos e nulos. "
        "Quando houver dados de pesquisa disponiveis, analise a preferência eleitoral por rede social. "
        "informe tambem as fontes, a data da pesquisa, o tamanho da amostra e a margem de erro."
    ),
    expected_output=(
        "Uma tabela organizada contendo candidato, intenção de voto, brancos/nulos, preferência por rede social, data da pesquisa e fontes. "
        "Um relatório organizado explicando os dados encontrados, as fontes e as limitações da pesquisa, sem recomendações ou julgamentos sobre candidatos."
    ),
    agent=pesquisador
)

equipe = Crew(
    agents=[pesquisador],
    tasks=[tarefa_pesquisa],
)

for tentativa in range(1, 4):

    try:
        resultado = equipe.kickoff()
        print(resultado)
        break
    
    except Exception as erro:
        print("Não foi possivel concluir a pesquisa.")
        print("Error:", erro)
        
        mensagem_erro = str(erro)
        
        if "429" in mensagem_erro:
            print("Limite de API atingido. Novas tentativas foram interrompidas.")
            break
        
        elif "503" in mensagem_erro:
            tempo_espera = 5 * tentativa
            
            if tentativa < 3:
                print(f"Tentativa {tentativa} falhou. Tentando novamente em {tempo_espera} segundos...")
                time.sleep(tempo_espera)
            else:
                print("Número máximo de tentativa atingido.")
                break
            
        else:
            print("Ocorreu um erro diferente de 429 e 503.")
            break
        