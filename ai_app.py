from fastapi import FastAPI

from ai_contract import router

app = FastAPI(
    title="Project Archive AI Server",
    description="Spring 백엔드가 호출하는 임베딩·RAG·요약·면접 API",
    version="1.0.0",
)
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ai_app:app", host="0.0.0.0", port=8000, reload=True)
