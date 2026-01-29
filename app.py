import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from loaders import *
import tempfile
from langchain.prompts import ChatPromptTemplate

tipos_arquivos_validos = ['Site','Youtube','PDF','CSV','TXT']
config_modelos = {'OpenAI':
                    {'modelos':['gpt-4o-mini','GPT-5 mini','GPT-5 nano'],
                     'chat':ChatOpenAI},
                     
                'Groq':
                    {'modelos':['llama-3.3-70b-versatile','openai/gpt-oss-20b'],
                     'chat':ChatGroq}
                  
                  }

MEMORIA = ConversationBufferMemory()


def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo=='Site':
        documento = carrega_site(arquivo)
    if tipo_arquivo=='Youtube':
        documento = carrega_youtube(arquivo)
    if tipo_arquivo=='PDF':
        with tempfile.NamedTemporaryFile(suffix='.pdf',delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp= temp.name
        documento = carrega_pdf(nome_temp)
    if tipo_arquivo=='CSV':
        with tempfile.NamedTemporaryFile(suffix='.csv',delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp= temp.name
        documento = carrega_pdf(nome_temp)
    if tipo_arquivo=='TXT':
        with tempfile.NamedTemporaryFile(suffix='.txt',delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp= temp.name
        documento = carrega_pdf(nome_temp)
    return documento

def carrega_modelo(provedor,modelo, api_key, tipo_arquivo, arquivo):    
    documento = carrega_arquivos(tipo_arquivo,arquivo)
    system_message = '''Você é um assistente amigável chamado Oráculo.
    Você possui acesso às seguintes informações vindas 
    de um documento {}: 

    ####
    {}
    ####

    Utilize as informações fornecidas para basear as suas respostas.

    Sempre que houver $ na sua saída, substita por S.

    Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" 
    sugira ao usuário carregar novamente o Oráculo!
    '''.format(tipo_arquivo, documento)

    print(system_message)
    
    template = ChatPromptTemplate.from_messages(
        [
            ('system',system_message),
            ('placeholder','{chat_history}'),
            ('user','{input}')
        ]
    )
    chat = config_modelos[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat

    st.session_state['chain'] = chain
    


def pagina_chat():
    st.header('🔮 Welcome to the Oracle', divider = True)

    chain = st.session_state.get('chain')

    if chain is None:
        st.error('Carrege o Oráculo')
        st.stop()

    memoria = st.session_state.get('memoria', MEMORIA)
    
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input('Talk to Oracle')
    if input_usuario:
        chat = st.chat_message('human')
        chat.markdown(input_usuario)

        chat = st.chat_message('ai')
        resposta = chat.write_stream(chain.stream({
            'input':input_usuario,
            'chat_history': memoria.buffer_as_messages
            }))
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria


def sidebar():
    tabs = st.tabs(['Upload de arquivos','Seleção de modelos'])
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo',tipos_arquivos_validos)
        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a url do Site')
        if tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a url do video')
        if tipo_arquivo == 'PDF':
            arquivo = st.file_uploader('Faça o upload do arquivo pdf', type = ['.pdf'])
        if tipo_arquivo == 'CSV':
            arquivo = st.file_uploader('Faça o upload do arquivo pdf', type = ['.csv'])
        if tipo_arquivo == 'TXT':
            arquivo = st.file_uploader('Faça o upload do arquivo pdf', type = ['.txt'])

    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor do modelo', config_modelos.keys())
        modelo = st.selectbox('Selecione o modelo', config_modelos[provedor]['modelos'])
        api_key = st.text_input(
            f'Adicione a api key do provedor {provedor}',
            value=st.session_state.get(f'api_key_{provedor}'))

        st.session_state[f'api_key_{provedor}'] = api_key

    if st.button('Inicializar oráculo', use_container_width=True):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)

    if st.button('Apagar histórico de conversa', use_container_width=True):
        st.session_state['memoria']=MEMORIA
def main():
    with st.sidebar:
        sidebar()
    pagina_chat()


if __name__ == '__main__':
    main()