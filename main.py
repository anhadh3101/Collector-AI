from fastapi import FastAPI
import logging

app = FastAPI()

@app.get("/")
async def root():
    # Debug message
    print(f"The type of the returned message is: {type({})}")
    return {"message": "Hello World"}

