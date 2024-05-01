import yaml
from operator import itemgetter
from typing import Dict, List, Optional, Sequence, Literal

from langchain_community.embeddings.yandex import YandexGPTEmbeddings
from langchain_community.chat_models import ChatYandexGPT
from langchain_community.chat_models import GigaChat
from langchain_community.llms.yandex import YandexGPT
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate,ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnablePassthrough, 
    RunnableLambda,
    RunnableBranch,
    Runnable,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import LanguageModelLike

from langchain_community.chat_message_histories import DynamoDBChatMessageHistory

import os
import boto3


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, str]]]


def get_first_document(docs):
    return {
            'content': docs[0].metadata['text'],
            'additional_kwargs': {},
            'type': 'ai',
            'name': None,
            'example': False
        }


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def print_me(input):
    print("Printing!!!", input)
    return input

# Reading a prompt for a language model
prompts_path = os.path.join(os.getcwd(), 'prompts/prompts.yaml')
with open(prompts_path, "r", encoding='utf-8') as f:
    prompts = yaml.safe_load(f)


class FAQSystem:
    def __init__(self, api_key, folder_id, session_id):
        self.api_key = api_key
        self.folder_id = folder_id
        self.embeddings = YandexGPTEmbeddings(
            api_key=api_key, 
            folder_id=folder_id, 
            sleep_interval=0.1
        )
        self.k = 4
        self.session_id = session_id
        self.boto3_session = boto3.Session(
            aws_access_key_id=os.getenv('STATIC_KEY_ID'),
            aws_secret_access_key=os.getenv('STATIC_KEY_SECRET'),
            region_name='ru-central1'
        )
        self.history = DynamoDBChatMessageHistory(
            boto3_session=self.boto3_session,
            table_name="SessionTable", 
            session_id=self.session_id,
            endpoint_url="http://127.0.0.1:8000/", 
        )   


    def get_retriever(self) -> BaseRetriever:
        vectorstore = Chroma(
            persist_directory="./chroma_db", 
            embedding_function=self.embeddings
        )
        return vectorstore.as_retriever(search_kwargs=dict(k=4), search_type="mmr")

    
    def create_retriever_chain(
        self, llm: LanguageModelLike, retriever: BaseRetriever,
    ) -> Runnable:
        
        def choose_route(info):
            if "dff" in info['topic'].lower():
                return itemgetter("condense_question") | retriever
            else:
                message = "Давайте поговорим о деле"
                return message

        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(prompts['rephrase_template'])
        condense_question_chain = RunnablePassthrough.assign(
            condense_question=CONDENSE_QUESTION_PROMPT | llm | StrOutputParser(),
        ).with_config(
            run_name="CondenseQuestion",
        )
        ROUTE_PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system", prompts['route_template']),
                ("human", "{condense_question}"),
            ]
        )
        topic_chain = RunnablePassthrough.assign(
            condense_question=itemgetter("condense_question"),
            topic=ROUTE_PROMPT | llm,
        ).with_config(
            run_name="GetTopic",
        )
        conversation_chain_with_history = condense_question_chain | topic_chain | RunnableLambda(choose_route)
        conversation_chain_with_no_history = topic_chain | RunnableLambda(choose_route)
        return RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),
                conversation_chain_with_history.with_config(run_name="RetrievalChainWithHistory"),
            ),
            (
                RunnablePassthrough.assign(condense_question=itemgetter("question")).with_config(
                    run_name="Itemgetter:question"
                )
                | conversation_chain_with_no_history
            ).with_config(run_name="RetrievalChainWithNoHistory"),
        ).with_config(run_name="RouteDependingOnChatHistory")
        
    
    def get_relevant_documents(self, query):

        docs_and_scores = self.vectorstore.similarity_search_with_score(query=query, k=self.k)
        result = []
        if docs_and_scores:
            max_score = docs_and_scores[0][1]
            if max_score >= self.score_threshold:
                docs_and_scores = docs_and_scores[:1]
            for doc, score in docs_and_scores:
                doc.metadata['score'] = score
                result.append(doc)
                
        return result
    
    
    def process_query(self):

        model_retriever = YandexGPT(
            folder_id=self.folder_id, 
            api_key=self.api_key, 
            temperature=0, 
            model_name='yandexgpt',
            sleep_interval=1.0
        )

        model_synthesizer = YandexGPT(
            folder_id=self.folder_id, 
            api_key=self.api_key, 
            temperature=0.2, 
            model_name='yandexgpt',
            sleep_interval=1.0
        )
    
        retriever = self.get_retriever()

        retriever_chain = self.create_retriever_chain(
            model_retriever,
            retriever,
        ).with_config(run_name="FindDocs")

        context = (
            RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
            .with_config(run_name="RetrieveDocs")
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts['response_template']),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        response_synthesizer = prompt | model_synthesizer

        chain = (
            context
            | response_synthesizer
        )

        def route_retriever_output(info):
            retriever_output = info['docs']
            if isinstance(retriever_output, str):
                return retriever_output
            else:
                return chain
        
        chain_with_message_history = RunnableWithMessageHistory(
            RunnablePassthrough.assign(docs=retriever_chain) | RunnableLambda(route_retriever_output),
            lambda session_id: self.history,
            input_messages_key="question",
            history_messages_key="history",
            config={"configurable": {"session_id": self.session_id}}
        )

        def trim_messages(chain_input):
            stored_messages = self.history.messages
            if len(stored_messages) <= 2:
                return False

            self.history.clear()

            for message in stored_messages[-2:]:
                self.history.add_message(message)

            return True 

        chain_with_trimming = (
            RunnablePassthrough.assign(messages_trimmed=trim_messages)
            | chain_with_message_history
        )

        return chain_with_trimming