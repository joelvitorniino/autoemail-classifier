from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import hashlib
import time
from app.email_processor import EmailProcessor

# Initialize FastAPI app
app = FastAPI()

# Configure templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
email_processor = EmailProcessor()

# Constants
PROCESSING_TIME_ESTIMATES = {
    "fast": lambda n: 2 * n + 1,
    "balanced": lambda n: 5 * n + 3,
    "thorough": lambda n: 10 * n + 5
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main index page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-email")
async def process_email(
    text: str = Form(None),
    file: UploadFile = File(None),
    mode: str = Form("fast")
) -> JSONResponse:
    """
    Process email content either from direct text input or file upload.
    
    Args:
        text: Raw email text content
        file: Uploaded file containing email content
        mode: Processing mode ('fast', 'balanced', 'thorough')
    
    Returns:
        JSONResponse containing processing results and metadata
    """
    start_time = time.time()
    
    # Validate input
    content, is_file = await _get_content_from_input(text, file)
    if not content:
        return JSONResponse(
            {"error": "No text or file provided."},
            status_code=400
        )

    # Process messages
    messages = [msg.strip() for msg in content.split('---') if msg.strip()]
    results = await email_processor.process_multiple_messages(content, mode)
    
    # Calculate processing metrics
    processing_metrics = _calculate_processing_metrics(
        start_time=start_time,
        num_messages=len(messages),
        mode=mode
    )
    
    return JSONResponse({
        "results": results,
        "processing_info": processing_metrics
    })

async def _get_content_from_input(
    text: str,
    file: UploadFile
) -> tuple[str, bool]:
    """Extract content from either text or file input."""
    if file:
        return (await file.read()).decode("utf-8"), True
    if text:
        return text, False
    return None, False

def _calculate_processing_metrics(
    start_time: float,
    num_messages: int,
    mode: str
) -> dict:
    """Calculate processing time metrics."""
    estimated_time = PROCESSING_TIME_ESTIMATES.get(
        mode,
        PROCESSING_TIME_ESTIMATES["fast"]
    )(num_messages)
    
    actual_time = time.time() - start_time
    
    return {
        "estimated_time": round(estimated_time, 2),
        "actual_time": round(actual_time, 2),
        "num_messages": num_messages,
        "mode": mode
    }