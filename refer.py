import streamlit as st
import tiktoken
import getpass
import os

from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory


# 랭스미스api설정
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ["LANGCHAIN_PROJECT"] = "loader"

def main():
    st.set_page_config(
    page_title="DirChat",
    page_icon=":books:")

    st.title("_Private Data :red[QA Chat]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None


    
    with st.sidebar:
                
        langsmith_api_key = st.text_input("LangSmith API Key 입력:", key="langsmith_api_key")
        os.environ['LANGCHAIN_API_KEY'] = st.session_state.get("langsmith_api_key")
        st.success("LangSmith API Key가 저장되었습니다.")
                
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        if uploaded_files:
                files_text = get_text(uploaded_files)
                text_chunks = get_text_chunks(files_text)
                vetorestore = get_vectorstore(text_chunks)
                     
                # Conversation chain 초기화
                openai_api_key = st.secrets["openai_api_key"]
                        
                st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 
                st.session_state.processComplete = True
                st.success("문서 처리가 완료되었습니다!")

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):

                # 검색 결과와 점수 가져오기
                source_documents = get_documents_with_scores(vetorestore, query, top_k=5)

                # Chat 처리
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']

                # 응답 출력
                st.markdown(response)
                
                # 참고 문서 출력
                display_relevant_documents(source_documents, threshold=0.3)
                
        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
                
                
                
                # result = chain({"question": query})
                # with get_openai_callback() as cb:
                #     st.session_state.chat_history = result['chat_history']
                # response = result['answer']
                # source_documents = result['source_documents']
                # st.markdown(response)
                
                # for doc in source_documents:
                #     st.write(doc.metadata)
                    
                # # 참고 문서 필터링 및 표시
                # if source_documents:
                #     threshold = 0.8  # 연관성 기준
                #     filtered_documents = [
                #         doc for doc in source_documents if doc.metadata.get("score", 1.0) >= threshold
                #     ]
                
                #     if not filtered_documents:
                #         st.info("관련성 있는 참고 문서를 찾을 수 없습니다.")
                #     else:
                #         for doc in filtered_documents[:3]:  # 최대 3개까지만 표시
                #             source = doc.metadata.get("source", "출처 알 수 없음")
                #             page = doc.metadata.get("page", "알 수 없음")
                #             with st.expander(f"참고 문서: {source} (페이지: {page})"):
                #                 st.markdown(doc.page_content)
                    
                    
#                    filtered_documents = [
#                        doc for doc in source_documents if doc.metadata.get("score", 0) >= threshold
#                    ]
#                    
#                    for doc in filtered_documents[:3]:  # 최대 3개의 문서 표시
#                        source = doc.metadata.get("source", "출처 알 수 없음")
#                        page = doc.metadata.get("page", "알 수 없음")
#                        
#                        with st.expander("참고 문서 확인"):
#                        # 중복 방지: source에 "Page" 정보가 이미 포함된 경우 page를 따로 표시하지 않음
#                            if "Page" in source:
#                                st.markdown(f"**출처:** {source}", help=doc.page_content)
#                            else:
#                                st.markdown(f"**출처:** {source}, **Page ** {page}", help=doc.page_content)
                
#                with st.expander("참고 문서 확인"):
#                    for doc in source_documents[:3]:
#                        source = doc.metadata.get("source", "출처 알 수 없음")
#                        page = doc.metadata.get("page", "알 수 없음")
                        
                        # 중복 방지: source에 "Page" 정보가 이미 포함된 경우 page를 따로 표시하지 않음
#                        if "Page" in source:
#                            st.markdown(f"**출처:** {source}", help=doc.page_content)
#                        else:
#                            st.markdown(f"**출처:** {source}, **Page ** {page}", help=doc.page_content)
                    
                  

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
            # 페이지 번호를 메타데이터에 보정하여 추가
            for doc in documents:
                page = doc.metadata.get("page", "알 수 없음")
                if isinstance(page, int):
                    page += 1  # 페이지 번호 보정
                doc.metadata["source"] = f"{file_name}, Page {page}"  # 페이지 포함
            #doc_list.extend(documents)
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks
            
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["openai_api_key"])
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


def get_documents_with_scores(vetorestore, query, top_k=5):
    # similarity_search_with_score로 검색 결과와 점수 가져오기
    results_with_scores = vetorestore.similarity_search_with_score(query, k=top_k)

    # 점수를 metadata에 추가
    documents = []
    for doc, score in results_with_scores:
        doc.metadata["score"] = score  # 점수를 metadata에 추가
        documents.append(doc)
    
    return documents

def display_relevant_documents(source_documents, threshold=0.8):
    # 점수를 기준으로 필터링
    filtered_documents = [
        doc for doc in source_documents if doc.metadata.get("score", 1.0) >= threshold
    ]

    if not filtered_documents:
        st.info("관련성 있는 참고 문서를 찾을 수 없습니다.")
    else:
        for doc in filtered_documents[:3]:  # 최대 3개까지만 표시
            source = doc.metadata.get("source", "출처 알 수 없음")
            page = doc.metadata.get("page", "알 수 없음")

            if "Page" in source:
                with st.expander(f"참고 문서: {source} "):
                    st.markdown(doc.page_content)
            else:
                with st.expander(f"참고 문서: {source} ,Page {page}"):
                    st.markdown(doc.page_content)
                
            

def get_conversation_chain(vetorestore,openai_api_key):
    
    prompt_template = """
    당신은 마케팅 전문가로, 사용자가 묻는 질문에 대해 문서와 웹을 기반으로 답변합니다. 
    사용자가 질문한 내용이 문서에 포함되어 있다면 문서를 최우선으로 참조하세요.
    만약 문서에 포함되어 있지 않은 경우, 신뢰할 수 있는 웹 데이터를 기반으로 답변을 작성하세요.
    모든 답변은 마케팅 관점에서 구체적이고 전략적인 인사이트를 제공하도록 작성하세요.

    문서를 기반으로 답변한 경우에는 반드시 "문서를 참조했습니다"라고 명시하고, 
    자체 데이터를 참조한 경우 "자체 데이터를 참조했습니다"라고 명시하십시오.

    # 문서 내용:
    {context}

    # 질문:
    {question}

    # 답변:
    """
        
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    
    
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-4o',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            combine_docs_chain_kwargs={"prompt": custom_prompt}, 
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()
