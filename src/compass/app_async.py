import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

from config.settings import load_settings
from agents.chatbot_core import ChatbotCore
from api.slack_bot_async import AsyncSlackChatbot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app_handler = None
handler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_handler, handler
    logger.info("Starting application initialization...")
    
    try:
        settings = load_settings()
        logger.info("Settings loaded.")
        
        # Initialize Core (DB connections, VertexAI, etc)
        # ChatbotCore.create is an async factory method
        logger.info("Initializing ChatbotCore...")
        core = await ChatbotCore.create(settings)
        logger.info("ChatbotCore initialized.")
        
        # Initialize Bolt App
        logger.info("Initializing AsyncSlackChatbot...")
        app_handler = AsyncSlackChatbot(settings, core)
        handler = AsyncSlackRequestHandler(app_handler)
        logger.info("AsyncSlackChatbot initialized.")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        raise e
    
    yield
    
    # Cleanup logic can go here (e.g., closing DB connections)
    logger.info("Shutting down application...")

api = FastAPI(lifespan=lifespan)

@api.post("/slack/events")
async def endpoint(req: Request):
    """
    Endpoint principal para recibir eventos de Slack.
    """
    if handler is None:
        return {"error": "App not initialized"}, 503
    return await handler.handle(req)

@api.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}
