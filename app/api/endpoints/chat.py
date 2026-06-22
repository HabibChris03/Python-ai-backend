import json
import logging
import re

from app.core.limiter import limiter
from app.core.timer import timed
from app.models.schemas import ChatbotResponse
from app.services.ai_models import ai_models
from app.services.chatbot_service import (
    search_user_documents,
    check_found_documents,
    get_document_expiry_info,
)
from fastapi import APIRouter, Form, HTTPException, Request
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_CONTEXT_LEN = 500
MAX_MESSAGE_LEN = 1000
INJECTION_PATTERNS = [
    "ignore previous",
    "ignore all instructions",
    "system:",
    "jailbreak",
    "you are now",
]


@router.post("/chat", response_model=ChatbotResponse)
@limiter.limit("10/minute")
async def chat_with_bot(
    request: Request,
    message: str = Form(...),
    context: str = Form(""),
    user_id: str = Form(None),
):
    context = (context or "")[:MAX_CONTEXT_LEN].strip()
    message = message[:MAX_MESSAGE_LEN].strip()

    combined = (message + " " + context).lower()
    if any(p in combined for p in INJECTION_PATTERNS):
        raise HTTPException(status_code=400, detail="Invalid input")

    # ── Groq LLM path with database access ────────────────────────────────────
    if ai_models.llm and user_id:
        try:
            # Pre-fetch database info if user is asking about their documents
            db_context = ""
            if any(
                keyword in message.lower()
                for keyword in ["my document", "found", "check", "search", "expire", "expir"]
            ):
                # Get user's documents
                doc_search = await search_user_documents(user_id, message)
                if doc_search.get("documents"):
                    db_context += (
                        "\n\nUSER'S DOCUMENTS:\n"
                        + json.dumps(doc_search["documents"], indent=2, default=str)
                    )
                
                # Check found documents if searching
                if "found" in message.lower():
                    found = await check_found_documents(message)
                    if found.get("documents"):
                        db_context += (
                            "\n\nFOUND DOCUMENTS:\n"
                            + json.dumps(found["documents"], indent=2, default=str)
                        )
                
                # Get expiry info
                expiry = await get_document_expiry_info(user_id)
                if expiry.get("total"):
                    db_context += (
                        "\n\nDOCUMENT EXPIRY STATUS:\n"
                        + json.dumps(expiry, indent=2, default=str)
                    )

            chat_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        f"""You are DocBot, the intelligent assistant for the DocuSecure (Smart Document Recovery System).

STRICT CONSTRAINTS:
1. ONLY discuss topics related to document management, scanning, security, encryption, and the DocuSecure application features.
2. If the user asks about anything else (politics, sports, general trivia, etc.), politely decline and steer the conversation back to documents or app support.
3. Provide helpful, concise, and professional advice on document safety and app usage.
4. You have access to the user's personal document database. Answer questions about their documents based on the data provided.
5. When a user asks if their document has been found, search the found-documents database and inform them.

APP KNOWLEDGE:
- Features: Biometric vault, AES-256 encryption, AI recognition, document tracking.
- Support: Users can scan IDs, passports, and certificates to keep them safe.
- Found Documents: If a document is marked as "found", it's available in the public found-items database.

DATABASE ACCESS:
You can query the user's documents and the found-items database to answer questions about:
- "Has my document been found?"
- "Do I have a passport on file?"
- "When does my ID expire?"
- "Show me my documents"
- "What documents do I have?"

{db_context}
""",
                    ),
                    ("user", "Context: " + context + "\n\nUser Query: " + message),
                ]
            )

            chain = chat_prompt | ai_models.llm | StrOutputParser()

            with timed() as llm_timer:
                response = await chain.ainvoke({})

            logger.info("Groq response received in %.1f ms", llm_timer.elapsed_ms)

            return ChatbotResponse(
                response=response,
                intent="general",
                confidence=1.0,
                response_time_ms=round(llm_timer.elapsed_ms, 2),
                engine="groq",
            )
        except Exception:
            logger.exception("Groq error, falling back to rule-based response")

    # ── Local rule-based fallback ─────────────────────────────────────────────
    with timed() as fallback_timer:
        msg = message.lower()
        fallback_response = "I am currently operating in basic mode. "

        if "passport" in msg:
            fallback_response += "To manage your passport, go to the Documents tab. You can scan it to keep a secure digital copy."
        elif "scan" in msg or "add" in msg:
            fallback_response += "You can scan new documents by clicking the Scan button on the Home dashboard or the + button in the Documents section."
        elif "security" in msg or "safe" in msg or "encrypt" in msg:
            fallback_response += "DocuSecure uses AES-256 encryption. Your documents are encrypted on your device before being synced to your private vault."
        elif "lost" in msg or "find" in msg:
            fallback_response += "If you lost a document, check the Find tab to see if it has been registered in the national found-items database."
        elif "hi" in msg or "hello" in msg or "help" in msg:
            fallback_response += "I can help you with document scanning, security tips, and tracking lost IDs. What would you like to know?"
        else:
            fallback_response += "I am DocBot, your document assistant. I am having some trouble reaching my advanced brain (Groq), but I can still help with basic questions about scanning and document security!"

    return ChatbotResponse(
        response=fallback_response,
        intent="fallback",
        confidence=0.5,
        response_time_ms=round(fallback_timer.elapsed_ms, 2),
        engine="fallback",
    )
