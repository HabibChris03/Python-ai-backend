"""
Chatbot service with Node.js backend API access for document queries.
Provides tools for Groq to retrieve user documents and check their status via API calls.
"""

import logging
import httpx
from datetime import datetime
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

# HTTP client for Node.js backend requests
http_client = httpx.AsyncClient(timeout=30.0)


async def search_user_documents(
    user_id: str, query: str, doc_type: Optional[str] = None
) -> dict:
    """
    Search for documents belonging to the user via Node.js backend API.
    Returns document details if found.
    """
    try:
        # Call Node.js backend API to get user documents
        url = f"{settings.NODEJS_BACKEND_URL}/documents"
        headers = {"Authorization": f"Bearer {settings.API_KEY}"}
        
        response = await http_client.get(url, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Node.js API error: {response.status_code}")
            return {
                "success": False,
                "message": f"No documents found matching '{query}'.",
                "documents": [],
            }
        
        data = response.json()
        documents = data.get("documents", [])
        
        if not documents:
            return {
                "success": False,
                "message": f"No documents found matching '{query}'.",
                "documents": [],
            }
        
        # Filter documents based on query and type
        query_lower = query.lower()
        result_docs = []
        
        for doc in documents:
            # Check if document matches query
            name_match = query_lower in doc.get("name", "").lower()
            number_match = query_lower in doc.get("documentNumber", "").lower()
            
            if not (name_match or number_match):
                continue
                
            # Check document type filter
            if doc_type and doc.get("type", "").lower() != doc_type.lower():
                continue
            
            # Check expiry status
            is_expired = False
            expiry_date = doc.get("expiryDate")
            if expiry_date:
                try:
                    expiry_dt = datetime.fromisoformat(expiry_date.replace('Z', '+00:00'))
                    is_expired = expiry_dt < datetime.now()
                except:
                    pass
            
            result_docs.append(
                {
                    "id": doc.get("_id", ""),
                    "name": doc.get("name", ""),
                    "type": doc.get("type", ""),
                    "documentNumber": doc.get("documentNumber", ""),
                    "issueDate": doc.get("issueDate", ""),
                    "expiryDate": doc.get("expiryDate", ""),
                    "status": doc.get("status", ""),
                    "isExpired": is_expired,
                    "uploadedAt": doc.get("createdAt", ""),
                }
            )
            
            # Limit to 5 results
            if len(result_docs) >= 5:
                break
        
        if not result_docs:
            return {
                "success": False,
                "message": f"No documents found matching '{query}'.",
                "documents": [],
            }
        
        return {
            "success": True,
            "message": f"Found {len(result_docs)} document(s) matching '{query}'.",
            "documents": result_docs,
        }
    except Exception as e:
        logger.error(f"Error searching documents via API: {e}")
        return {
            "success": False,
            "message": "Error searching documents.",
            "documents": [],
        }


async def check_found_documents(query: str) -> dict:
    """
    Check if a document has been found and is available via Node.js backend API.
    Searches the public found-documents database.
    """
    try:
        # Call Node.js backend API to get found documents
        url = f"{settings.NODEJS_BACKEND_URL}/documents/found"
        
        response = await http_client.get(url)
        
        if response.status_code != 200:
            logger.error(f"Node.js API error: {response.status_code}")
            return {
                "success": False,
                "message": f"No found documents matching '{query}' in the database.",
            }
        
        data = response.json()
        documents = data.get("documents", data.get("data", []))
        
        if not documents:
            return {
                "success": False,
                "message": f"No found documents matching '{query}' in the database.",
            }
        
        # Filter documents based on query
        query_lower = query.lower()
        result_docs = []
        
        for doc in documents:
            # Check if document matches query
            name_match = query_lower in doc.get("name", "").lower()
            number_match = query_lower in doc.get("documentNumber", "").lower()
            
            if not (name_match or number_match):
                continue
            
            result_docs.append(
                {
                    "name": doc.get("name", ""),
                    "type": doc.get("type", ""),
                    "foundLocation": doc.get("foundLocation", ""),
                    "foundDate": doc.get("foundDate", ""),
                    "additionalNotes": doc.get("additionalNotes", ""),
                }
            )
            
            # Limit to 5 results
            if len(result_docs) >= 5:
                break
        
        if not result_docs:
            return {
                "success": False,
                "message": f"No found documents matching '{query}' in the database.",
            }
        
        return {
            "success": True,
            "message": f"Found {len(result_docs)} matching document(s) in the found-items database.",
            "documents": result_docs,
        }
    except Exception as e:
        logger.error(f"Error checking found documents via API: {e}")
        return {
            "success": False,
            "message": "Error checking found documents.",
        }


async def get_document_expiry_info(user_id: str) -> dict:
    """
    Get expiry status for all user documents via Node.js backend API.
    """
    try:
        from datetime import timedelta
        
        # Call Node.js backend API to get user documents
        url = f"{settings.NODEJS_BACKEND_URL}/documents"
        headers = {"Authorization": f"Bearer {settings.API_KEY}"}
        
        response = await http_client.get(url, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"Node.js API error: {response.status_code}")
            return {
                "total": 0,
                "expired": 0,
                "expiring_soon": [],
                "active": [],
            }
        
        data = response.json()
        documents = data.get("documents", [])
        
        if not documents:
            return {
                "total": 0,
                "expired": 0,
                "expiring_soon": [],
                "active": [],
            }
        
        expired = []
        expiring_soon = []
        active = []
        now = datetime.now()
        thirty_days = timedelta(days=30)
        
        for doc in documents:
            doc_info = {
                "name": doc.get("name", ""),
                "type": doc.get("type", ""),
                "expiryDate": doc.get("expiryDate", ""),
            }
            
            expiry_date = doc.get("expiryDate")
            if expiry_date:
                try:
                    expiry_dt = datetime.fromisoformat(expiry_date.replace('Z', '+00:00'))
                    if expiry_dt < now:
                        expired.append(doc_info)
                    elif expiry_dt < (now + thirty_days):
                        expiring_soon.append(doc_info)
                    else:
                        active.append(doc_info)
                except:
                    active.append(doc_info)
            else:
                active.append(doc_info)
        
        return {
            "total": len(documents),
            "expired": len(expired),
            "expiring_soon": expiring_soon,
            "active": active,
        }
    except Exception as e:
        logger.error(f"Error getting expiry info via API: {e}")
        return {
            "total": 0,
            "expired": 0,
            "expiring_soon": [],
            "active": [],
        }
