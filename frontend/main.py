import os
import json
import uuid
import asyncio
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    print("=" * 60)
    print("ERROR: No se encontro GEMINI_API_KEY en el archivo .env")
    print("=" * 60)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(BASE_DIR, "agents")
DATA_DIR = os.path.join(BASE_DIR, "data")
MEMORY_FILE = os.path.join(DATA_DIR, "memory.json")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

MODELS = {
    "gemini-2.5-flash-lite": {
        "name": "Gemini 2.5 Flash-Lite",
        "description": "Rapido, 1000 consultas/dia gratis.",
        "rpd": 1000
    },
    "gemini-2.5-flash": {
        "name": "Gemini 2.5 Flash",
        "description": "Equilibrio calidad/velocidad, 250 consultas/dia gratis.",
        "rpd": 250
    },
    "gemini-2.5-pro": {
        "name": "Gemini 2.5 Pro",
        "description": "El mas inteligente, 100 consultas/dia gratis.",
        "rpd": 100
    },
    "gemini-2.0-flash": {
        "name": "Gemini 2.0 Flash",
        "description": "Version anterior, 1500 consultas/dia gratis.",
        "rpd": 1500
    }
}

gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def load_memory() -> dict:
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_memory(memory: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def get_conversation(conversation_id: str) -> list:
    memory = load_memory()
    if conversation_id in memory:
        return memory[conversation_id]["messages"]
    return []


def save_message(conversation_id: str, agent_id: str, role: str, content: str):
    memory = load_memory()
    if conversation_id not in memory:
        memory[conversation_id] = {
            "agent_id": agent_id,
            "created_at": datetime.now().isoformat(),
            "messages": []
        }
    memory[conversation_id]["messages"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    save_memory(memory)


def load_agent_prompt(agent_id: str) -> str:
    filepath = os.path.join(AGENTS_DIR, f"{agent_id}.txt")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontro el agente '{agent_id}'. Verifica que exista agents/{agent_id}.txt"
        )


def list_agents() -> list:
    agents = []
    if not os.path.exists(AGENTS_DIR):
        os.makedirs(AGENTS_DIR)
        return agents
    for filename in sorted(os.listdir(AGENTS_DIR)):
        if filename.endswith(".txt"):
            agent_id = filename.replace(".txt", "")
            agents.append({
                "id": agent_id,
                "name": agent_id.replace("-", " ").replace("_", " ").title(),
                "filename": filename
            })
    return agents


async def call_gemini(model: str, system_prompt: str, history: list, user_message: str) -> str:
    if not gemini_client:
        raise HTTPException(status_code=500, detail="API key de Gemini no configurada.")

    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    try:
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=model,
            contents=contents,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.7,
                "max_output_tokens": 8192
            }
        )
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            raise HTTPException(status_code=429, detail="Limite de peticiones excedido. Espera o cambia de modelo.")
        raise HTTPException(status_code=500, detail=f"Error Gemini: {error_msg}")


app = FastAPI(title="Agentes IA")


@app.get("/api/agents")
async def api_list_agents():
    return {"agents": list_agents()}


@app.get("/api/models")
async def api_list_models():
    return {"models": MODELS}


@app.get("/api/conversations")
async def api_list_conversations():
    memory = load_memory()
    conversations = []
    for conv_id, conv_data in memory.items():
        conversations.append({
            "id": conv_id,
            "agent_id": conv_data.get("agent_id", "desconocido"),
            "created_at": conv_data.get("created_at", ""),
            "message_count": len(conv_data.get("messages", []))
        })
    return {"conversations": conversations}


@app.get("/api/conversations/{conversation_id}")
async def api_get_conversation(conversation_id: str):
    messages = get_conversation(conversation_id)
    return {"conversation_id": conversation_id, "messages": messages}


@app.delete("/api/conversations/{conversation_id}")
async def api_delete_conversation(conversation_id: str):
    memory = load_memory()
    if conversation_id in memory:
        del memory[conversation_id]
        save_memory(memory)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Conversacion no encontrada")


@app.post("/api/chat")
async def api_chat(request: dict):
    agent_id = request.get("agent_id")
    model = request.get("model", "gemini-2.5-flash")
    user_message = request.get("message")
    conversation_id = request.get("conversation_id")

    if not agent_id:
        raise HTTPException(status_code=400, detail="Falta agent_id")
    if not user_message:
        raise HTTPException(status_code=400, detail="Falta message")
    if model not in MODELS:
        raise HTTPException(status_code=400, detail=f"Modelo '{model}' no disponible.")

    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    system_prompt = load_agent_prompt(agent_id)
    history = get_conversation(conversation_id)
    save_message(conversation_id, agent_id, "user", user_message)

    response_text = await call_gemini(
        model=model,
        system_prompt=system_prompt,
        history=history,
        user_message=user_message
    )

    save_message(conversation_id, agent_id, "assistant", response_text)

    return {
        "response": response_text,
        "model": model,
        "agent_id": agent_id,
        "conversation_id": conversation_id
    }


@app.get("/api/status")
async def api_status():
    return {
        "status": "online",
        "gemini_configured": bool(GEMINI_API_KEY),
        "agents_count": len(list_agents()),
        "agents": [a["id"] for a in list_agents()]
    }


app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
