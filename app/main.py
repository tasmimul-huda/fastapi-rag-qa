from fastapi import FastAPI, Body, Request, Form
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from api.answer import answer_router
from api.upload import upload_router

import logging_config
from settings import Config

import warnings
warnings.filterwarnings("ignore")


conf = Config

def create_app():

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(answer_router) #, prefix="/api/v1"
    app.include_router(upload_router) #, prefix="/api/v1"

    return app


app = create_app()

