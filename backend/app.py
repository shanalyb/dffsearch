import os
from chain import FAQSystem
from fastapi import FastAPI
from dotenv import load_dotenv
from history.clickhouse import ClickHouseChatMessageHistory

load_dotenv()
API_KEY = os.getenv('API_KEY')
FOLDER_ID = "b1gjg18vo9sd3fk8qmus"


app = FastAPI(title="Messangers App")

@app.get("/message")
async def message(question: str, session_id: str, intent: str, node_name: str):
    ch_history = ClickHouseChatMessageHistory(
        host="158.160.119.89",
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

    uvicorn.run("app:app", host='10.128.0.27', port=8339, reload=True)
