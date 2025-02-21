from fastapi import FastAPI
from app.api.endpoints import users, items

app = FastAPI()

app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(items.router, prefix="/items", tags=["items"])

@app.get("/")
def home():
    return {"message": "Welcome to Pavement Identifier!"}
