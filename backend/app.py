import os
from chain import FAQSystem
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY')
FOLDER_ID = "b1gjg18vo9sd3fk8qmus"


app = FastAPI(title="Messangers App")

@app.get("/message")
async def message(question: str, session_id: str):
    faq_system = FAQSystem(API_KEY, FOLDER_ID, session_id)
    chain = faq_system.process_query()
    return chain.invoke({"question": question})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host='10.128.0.27', port=8339, reload=True)
