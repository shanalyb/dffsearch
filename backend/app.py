import os
from chain import FAQSystem
from fastapi import FastAPI
from dotenv import load_dotenv
from history.clickhouse import ClickHouseChatMessageHistory

load_dotenv()
API_KEY = os.getenv('API_KEY')
FOLDER_ID = "b1gjg18vo9sd3fk8qmus"


app = FastAPI(title="Messangers App")

@app.get("/stats")
async def stats(message: str, session_id: str, intent: str, node_name: str):
    print(message, session_id, intent, node_name)
    ch_history = ClickHouseChatMessageHistory(
        host="127.0.0.1",
        port=8123,
        table_name='dffserch_history',
        session_id=session_id,
    )
    ch_history.add_user_message(message, intent=intent, node_name=node_name)
    return 'OK'

@app.get("/message")
async def message(question: str, session_id: str, intent: str, node_name: str):
    ch_history = ClickHouseChatMessageHistory(
        host="127.0.0.1",
        port=8123,
        table_name='dffserch_history',
        session_id=session_id,
    )
    faq_system = FAQSystem(API_KEY, FOLDER_ID, session_id)
    chain = faq_system.process_query()
    answer = chain.invoke({"question": question})
    ch_history.add_user_message(question, intent=intent, node_name=node_name)
    ch_history.add_ai_message(answer, intent=intent, node_name=node_name)

    return answer


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host='127.0.0.1', port=8339, reload=True)
