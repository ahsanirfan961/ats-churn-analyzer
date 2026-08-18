from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.graph import close_graph
from app.config import CORS_ORIGINS
from app.routes import chat, threads


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_graph()


app = FastAPI(title="Churn Analyst Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(threads.router)


@app.get("/health")
def health():
    return {"status": "ok"}
