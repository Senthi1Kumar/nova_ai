import logging
import re
import json
import os
import httpx
from datetime import datetime
from pathlib import Path
from typing import Optional

# tiktoken is imported lazily inside count_tokens / limit_tokens so that
# importing this module doesn't fail when tiktoken isn't installed (it's
# only needed for token accounting, which the chat-app loop doesn't use).

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sanitize_llm_input(value: str, param_name: str = "parameter") -> str:
    """Light input sanitizer — strip control chars and quotes, keep spaces +
    standard punctuation. Caps length at 500 chars to bound prompt-injection
    surface.

    Note: the previous version stripped EVERYTHING except [a-zA-Z0-9,] which
    is fine for SQL identifier guards but destroys search queries (spaces and
    punctuation get removed). Web-search use-cases need the spaces back.
    """
    if not isinstance(value, str):
        return value
    # Drop ASCII control chars + quote chars; keep printable letters/digits/
    # whitespace/punctuation needed for natural-language queries.
    sanitized = re.sub(r"[\x00-\x1f\x7f\"'`]", "", value)
    # Collapse runs of whitespace.
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized[:500]


def sanitize_search_type(search_type: str, valid_types: list, default: str = "web") -> str:
    """
    Simplified: If 'news' in search_type (case-insensitive), return 'news'; else return 'web'.
    """
    search_type = sanitize_llm_input(search_type, "search_type")
    if "news" in search_type.lower():
        return "news"
    return "web"


def extract_search_metadata(search_data):
    """Extract and organize metadata from search results"""
    metadata = {
        'reliable_info': {
            'faq': [],
            'infobox': []
        },
        'less_reliable': {
            'discussions': []
        },
        'location_data': []
    }
    
    # Extract FAQ data
    if 'faq' in search_data and 'results' in search_data['faq']:
        for faq in search_data['faq']['results']:
            metadata['reliable_info']['faq'].append({
                'question': faq['question'],
                'answer': faq['answer'],
                'source': faq.get('title', '')
            })

    # Extract infobox data
    if 'infobox' in search_data and 'results' in search_data['infobox']:
        for box in search_data['infobox']['results']:
            info = {
                'title': box.get('title', ''),
                'description': box.get('description', ''),
                'long_desc': box.get('long_desc', ''),
                'attributes': box.get('attributes', []),
                'coordinates': box.get('coordinates', [])
            }
            metadata['reliable_info']['infobox'].append(info)

    # Extract discussions (marked as less reliable)
    if 'discussions' in search_data and 'results' in search_data['discussions']:
        for discussion in search_data['discussions']['results']:
            disc = {
                'title': discussion.get('title', ''),
                'description': discussion.get('description', ''),
                'data': discussion.get('data', {}),
                'age': discussion.get('age', ''),
                'source': discussion.get('meta_url', {}).get('hostname', '')
            }
            metadata['less_reliable']['discussions'].append(disc)

    # Extract and process location data
    if 'web' in search_data and 'results' in search_data['web']:
        for data_item in search_data['web']['results']:
            location = data_item.get("location")
            if location:
                # Clean up opening_hours
                opening_hours = location.get("opening_hours", {})
                cleaned_opening_hours = {}

                if "current_day" in opening_hours:
                    cleaned_opening_hours["current_day"] = [
                        {
                            "day": d.get("abbr_name"),
                            "opens": d.get("opens"),
                            "closes": d.get("closes"),
                        }
                        for d in opening_hours.get("current_day", [])
                    ]

                if "days" in opening_hours:
                    cleaned_opening_hours["days"] = [
                        {
                            "day": d[0].get("abbr_name"),
                            "opens": d[0].get("opens"),
                            "closes": d[0].get("closes"),
                        }
                        for d in opening_hours.get("days", [])
                        if d and isinstance(d, list) and len(d) > 0
                    ]

                filtered_location = {
                    "title": location.get("title"),
                    "url": location.get("url"),
                    "description": location.get("description"),
                    "family_friendly": location.get("family_friendly"),
                    "postal_address": location.get("postal_address"),
                    "opening_hours": cleaned_opening_hours,
                    "contact": location.get("contact"),
                    "pictures": [
                        pic.get("original")
                        for pic in location.get("pictures", {}).get("results", [])
                        if pic.get("original")
                    ],
                    "price_range": location.get("price_range"),
                    "rating": location.get("rating"),
                    "serves_cuisine": location.get("serves_cuisine"),
                }

                metadata['location_data'].append(filtered_location)

    return metadata


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken (lazy-imported)."""
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}, falling back to character/4 approximation")
        return len(text) // 4


def limit_tokens(text: str, max_tokens: int) -> str:
    """Limit the number of tokens in a given text (lazy-imported tiktoken)."""
    import tiktoken
    encoding = tiktoken.get_encoding("o200k_base")
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        return encoding.decode(tokens[:max_tokens])
    return text


def save_conversation_to_json(session, room_name: str, output_dir: str = "conversations"):
    """
    Save the conversation history to a JSON file.
    
    Args:
        session: The AgentSession instance containing the chat context
        room_name: The name of the room (used for filename)
        output_dir: Directory to save conversation files (default: "conversations")
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get the chat context from the session (history is a property, not a method)
    chat_ctx = session.history
    
    # Find first user message for filename and starting timestamp
    user_request = ""
    start_timestamp = None
    for item in chat_ctx.items:
        if item.type == "message" and item.role == "user":
            # Extract text content from message
            text_content = []
            for content in item.content:
                if isinstance(content, str):
                    text_content.append(content)
            user_request = " ".join(text_content) if text_content else ""
            # Use the first user message's timestamp as the conversation start time
            start_timestamp = item.created_at
            break
    
    # Sanitize user request for filename (first 100 chars)
    if user_request:
        # Remove special characters and keep only alphanumeric, spaces, hyphens, underscores
        sanitized = "".join(c if c.isalnum() or c in (' ', '-', '_') else '-' for c in user_request)
        # Replace multiple spaces/hyphens with single hyphen
        sanitized = "-".join(sanitized.split())
        # Limit to 100 characters
        sanitized = sanitized[:100]
    else:
        sanitized = "no-request"
    
    # Use start timestamp if available, otherwise use current time
    if start_timestamp:
        timestamp_dt = datetime.fromtimestamp(start_timestamp)
    else:
        timestamp_dt = datetime.now()
    timestamp_str = timestamp_dt.strftime("%Y%m%d_%H%M%S")
    
    # Convert chat items to JSON-serializable format
    conversation_data = {
        "room_name": room_name,
        "timestamp": datetime.now().isoformat(),
        "items": []
    }
    
    for item in chat_ctx.items:
        # Skip function call outputs
        item_dict = {
            "id": item.id,
            "type": item.type,
            "created_at": item.created_at,
        }
        
        # Handle different item types
        if item.type == "message":
            item_dict["role"] = item.role
            # Extract text content from message
            text_content = []
            for content in item.content:
                if isinstance(content, str):
                    text_content.append(content)
                elif hasattr(content, "type"):
                    if content.type == "image_content":
                        text_content.append(f"[Image: {content.id}]")
                    elif content.type == "audio_content":
                        text_content.append(f"[Audio: {content.transcript or 'No transcript'}]")
            item_dict["content"] = "\n".join(text_content) if text_content else ""
            item_dict["interrupted"] = item.interrupted
            if hasattr(item, "transcript_confidence") and item.transcript_confidence is not None:
                item_dict["transcript_confidence"] = item.transcript_confidence
                
        elif item.type == "function_call":
            item_dict["name"] = item.name
            item_dict["call_id"] = item.call_id
            item_dict["arguments"] = item.arguments
        
        conversation_data["items"].append(item_dict)
    
    # Generate filename with starting timestamp and sanitized user request
    filename = f"{timestamp_str}_{sanitized}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to JSON file (overwrites previous version with same filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    
    return filepath


def format_conversation_for_api(session) -> tuple[str, str]:
    """
    Format conversation data for API logging.
    Combines assistant responses with XML tags, ignoring function_call_output.
    
    Args:
        session: The AgentSession instance containing the chat context
    
    Returns:
        Tuple of (request_string, response_string)
        - request_string: Combined user messages
        - response_string: Combined assistant responses with XML tags for function calls and responses
    """
    chat_ctx = session.history
    
    request_parts = []
    response_parts = []
    
    for item in chat_ctx.items:
        # Skip function_call_output as requested
        if item.type == "function_call_output":
            continue
            
        if item.type == "message":
            if item.role == "user":
                # Extract text content from user message
                text_content = []
                for content in item.content:
                    if isinstance(content, str):
                        text_content.append(content)
                if text_content:
                    request_parts.append(" ".join(text_content))
                    
            elif item.role == "assistant":
                # Extract text content from assistant message
                text_content = []
                for content in item.content:
                    if isinstance(content, str):
                        text_content.append(content)
                if text_content:
                    response_parts.append(f"<response>{' '.join(text_content)}</response>")
                    
        elif item.type == "function_call":
            # Format function call with arguments
            func_name = item.name
            if hasattr(item, "arguments"):
                func_args = item.arguments
                # If arguments is a string (JSON), use it as-is; if dict, convert to JSON
                if isinstance(func_args, dict):
                    func_args = json.dumps(func_args)
                func_call_str = f"{func_name}({func_args})"
            else:
                func_call_str = func_name
            response_parts.append(f"<function_call>{func_call_str}</function_call>")
    
    request_string = "\n".join(request_parts) if request_parts else ""
    response_string = "\n".join(response_parts) if response_parts else ""
    
    return request_string, response_string


async def log_conversation_to_api(
    session,
    conversation_id: Optional[str] = None,
    message_id: Optional[str] = None,
    user_id: Optional[str] = None,
    model: Optional[str] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cost: float = 0.0,
    latency: float = 0.0,
    status: str = "completed",
    user_feedback: Optional[str] = None,
    api_url: Optional[str] = None,
) -> dict:
    """
    Log conversation to the observations API endpoint.
    
    Args:
        session: The AgentSession instance containing the chat context
        conversation_id: Unique identifier for the conversation
        message_id: Unique identifier for the message
        user_id: User identifier
        model: Model name used for the conversation
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_tokens: Total number of tokens
        cost: Cost of the conversation
        latency: Latency in seconds
        status: Status of the conversation (default: "completed")
        user_feedback: Optional user feedback
        api_url: API endpoint URL (defaults to environment variable or hardcoded)
    
    Returns:
        Dictionary with API response or error information
    """
    if api_url is None:
        api_url = os.getenv(
            "OBSERVATIONS_API_URL",
            "https://api4iresearcher-v5-1.elevatics.site/api/v1/observations/log"
        )
    
    # Validate session and history
    try:
        if not hasattr(session, "history"):
            raise ValueError("Session does not have history attribute")
        chat_ctx = session.history
        if not hasattr(chat_ctx, "items"):
            raise ValueError("Session history does not have items attribute")
    except Exception as e:
        error_msg = f"Invalid session data: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    
    # Format conversation data
    try:
        request_string, response_string = format_conversation_for_api(session)
    except Exception as e:
        error_msg = f"Error formatting conversation data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error": error_msg}
    
    # Generate IDs if not provided
    if conversation_id is None:
        conversation_id = chat_ctx.id if hasattr(chat_ctx, "id") else "unknown"
    if message_id is None:
        # Use the last item's ID if available
        items = chat_ctx.items
        message_id = items[-1].id if items else "unknown"
    
    # Prepare payload
    payload = {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "status": status,
        "request": request_string,
        "response": response_string,
        "model": model or "unknown",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
        "latency": latency,
        "user_id": user_id or "unknown",
        "user_feedback": user_feedback or "",
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                api_url,
                json=payload,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                }
            )
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                except Exception:
                    response_data = {"status": "success", "message": "No JSON response"}
                logger.info(f"Successfully logged conversation to API: {conversation_id}")
                return {"success": True, "data": response_data}
            else:
                error_msg = f"API error: {response.status_code} - {response.text[:500]}"
                logger.error(f"Failed to log conversation to API: {error_msg}")
                return {"success": False, "error": error_msg}
                
    except Exception as e:
        error_msg = f"Exception logging conversation to API: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}