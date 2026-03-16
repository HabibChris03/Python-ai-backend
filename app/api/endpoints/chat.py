from fastapi import APIRouter, Form, HTTPException
import logging
from app.models.schemas import ChatbotResponse
from app.services.ai_models import ai_models
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/chat", response_model=ChatbotResponse)
async def chat_with_bot(message: str = Form(...), context: str = Form("")):
    # 1. Try OpenAI if initialized
    if ai_models.llm:
        try:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are DocBot, an AI assistant for document management. Provide helpful, concise responses."),
                ("user", f"Context: {context}\n\nUser Query: {message}")
            ])
            
            chain = chat_prompt | ai_models.llm | StrOutputParser()
            response = await chain.ainvoke({})
            
            return ChatbotResponse(
                response=response,
                intent="general",
                confidence=1.0
            )
        except Exception as e:
            logger.error(f"OpenAI Error (Falling back to local rules): {e}")
            # Continue to fallback logic below if OpenAI fails (e.g., quota exceeded)
    
    # 2. Local Fallback Logic (Rule-based)
    msg = message.lower()
    fallback_response = "I'm currently operating in basic mode. "
    
    if "passport" in msg:
        fallback_response += "To manage your passport, go to the Documents tab. You can scan it to keep a secure digital copy."
    elif "scan" in msg or "add" in msg:
        fallback_response += "You can scan new documents by clicking the 'Scan' button on the Home dashboard or the '+' button in the Documents section."
    elif "security" in msg or "safe" in msg or "encrypt" in msg:
        fallback_response += "DocuSecure uses AES-256 encryption. Your documents are encrypted on your device before being synced to your private vault."
    elif "lost" in msg or "find" in msg:
        fallback_response += "If you lost a document, check the 'Find' tab to see if it has been registered in the national found-items database."
    elif "hi" in msg or "hello" in msg or "help" in msg:
        fallback_response += "I can help you with document scanning, security tips, and tracking lost IDs. What would you like to know?"
    else:
        fallback_response += "I'm DocBot, your document assistant. I'm having some trouble reaching my advanced brain (OpenAI), but I can still help with basic questions about scanning and document security!"

    return ChatbotResponse(
        response=fallback_response,
        intent="fallback",
        confidence=0.5
    )
