from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import hashlib
import time
from app.email_processor import EmailProcessor

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

email_processor = EmailProcessor()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-email")
async def process_email(
    text: str = Form(None),
    file: UploadFile = File(None),
    mode: str = Form("fast")  # Adiciona o parâmetro mode com valor padrão
):
    start_time = time.time()
    
    if file:
        content = (await file.read()).decode("utf-8")
        is_file = True
    elif text:
        content = text
        is_file = False
    else:
        return JSONResponse({"error": "Nenhum texto ou arquivo enviado."}, status_code=400)

    # Estima o tempo de processamento com base no modo
    num_messages = len([msg for msg in content.split('---') if msg.strip()])
    estimated_times = {
        "fast": 2 * num_messages + 1,
        "balanced": 5 * num_messages + 3,
        "thorough": 10 * num_messages + 5
    }
    estimated_time = estimated_times.get(mode, 2 * num_messages + 1)
    
    # Processa as mensagens usando a instância com o modo especificado
    results = await email_processor.process_multiple_messages(content, mode)
    
    # Calcula o tempo real de processamento
    processing_time = time.time() - start_time
    
    return JSONResponse({
        "results": results,
        "processing_info": {
            "estimated_time": round(estimated_time, 2),
            "actual_time": round(processing_time, 2),
            "is_file": is_file,
            "num_messages": len(results),
            "mode": mode
        }
    })