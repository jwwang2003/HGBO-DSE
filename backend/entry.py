# api_server.py

import uuid
import asyncio
import secrets
from datetime import datetime, timedelta
import os
from typing import Optional, Dict

from fastapi import (
    FastAPI, Depends, HTTPException, status, Header,
    WebSocket, WebSocketDisconnect, BackgroundTasks, Query
)
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column, String, DateTime, create_engine, Integer, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --- Database setup ---
DATABASE_URL = "sqlite:///./api_keys.db"
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class APIKey(Base):
    __tablename__ = "api_keys"
    id         = Column(Integer, primary_key=True, index=True)
    key        = Column(String, unique=True, index=True, nullable=False)
    memo       = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)   # ← new expiry

Base.metadata.create_all(bind=engine)


# --- FastAPI app and security ---
app = FastAPI()
security = HTTPBasic()

ADMIN_USER_ENV = "HGBO_ADMIN_USER"
ADMIN_PASS_ENV = "HGBO_ADMIN_PASS"

def get_admin_creds(creds: HTTPBasicCredentials = Depends(security)):
    admin_user = os.getenv(ADMIN_USER_ENV)
    admin_pass = os.getenv(ADMIN_PASS_ENV)
    if not admin_user or not admin_pass:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin credentials are not configured",
        )
    if not (secrets.compare_digest(creds.username, admin_user)
            and secrets.compare_digest(creds.password, admin_pass)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return creds.username

# serve static admin UI
if not os.path.isdir("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/admin", response_class=HTMLResponse, dependencies=[Depends(get_admin_creds)])
def admin_ui():
    return FileResponse("static/admin.html")


# --- Pydantic schemas ---
class KeyCreate(BaseModel):
    memo: Optional[str] = Field(None, max_length=500)
    expire_days: Optional[int] = Field(30, gt=0)

class KeyUpdate(BaseModel):
    memo: str = Field(..., max_length=500)

class KeyOut(BaseModel):
    key: str
    created_at: datetime
    memo: Optional[str] = None
    expires_at: datetime

class DataIn(BaseModel):
    payload: str = Field(..., min_length=1, max_length=1000)

class TaskStart(BaseModel):
    work_units: int = Field(..., gt=0, lt=101)


# --- Admin endpoints ---
@app.post(
    "/admin/keys",
    response_model=KeyOut,
    dependencies=[Depends(get_admin_creds)]
)
def create_key(key_in: KeyCreate):
    db = SessionLocal()
    now = datetime.utcnow()
    new = APIKey(
        key=uuid.uuid4().hex,
        memo=key_in.memo,
        created_at=now,
        expires_at=now + timedelta(days=key_in.expire_days)
    )
    db.add(new)
    db.commit()
    db.refresh(new)
    db.close()
    return KeyOut(
        key=new.key,
        created_at=new.created_at,
        memo=new.memo,
        expires_at=new.expires_at
    )

@app.get(
    "/admin/keys",
    response_model=list[KeyOut],
    dependencies=[Depends(get_admin_creds)]
)
def list_keys():
    db = SessionLocal()
    keys = db.query(APIKey).all()
    db.close()
    return [
        KeyOut(
          key=k.key,
          created_at=k.created_at,
          memo=k.memo,
          expires_at=k.expires_at
        )
        for k in keys
    ]

@app.patch(
    "/admin/keys/{api_key}/memo",
    response_model=KeyOut,
    dependencies=[Depends(get_admin_creds)]
)
def update_memo(api_key: str, key_upd: KeyUpdate):
    db = SessionLocal()
    obj = db.query(APIKey).filter(APIKey.key == api_key).first()
    if not obj:
        db.close()
        raise HTTPException(status_code=404, detail="Key not found")
    obj.memo = key_upd.memo
    db.commit()
    db.refresh(obj)
    db.close()
    return KeyOut(
        key=obj.key,
        created_at=obj.created_at,
        memo=obj.memo,
        expires_at=obj.expires_at
    )

@app.patch(
    "/admin/keys/{api_key}/revoke",
    response_model=KeyOut,
    dependencies=[Depends(get_admin_creds)]
)
def revoke_key(api_key: str):
    db = SessionLocal()
    obj = db.query(APIKey).filter(APIKey.key == api_key).first()
    if not obj:
        db.close()
        raise HTTPException(status_code=404, detail="Key not found")
    obj.expires_at = datetime.utcnow()
    db.commit()
    db.refresh(obj)
    db.close()
    return KeyOut(
        key=obj.key,
        created_at=obj.created_at,
        memo=obj.memo,
        expires_at=obj.expires_at
    )

@app.patch(
    "/admin/keys/{api_key}/refresh",
    response_model=KeyOut,
    dependencies=[Depends(get_admin_creds)]
)
def refresh_key(
    api_key: str,
    days: int = Query(30, gt=0)
):
    db = SessionLocal()
    obj = db.query(APIKey).filter(APIKey.key == api_key).first()
    if not obj:
        db.close()
        raise HTTPException(status_code=404, detail="Key not found")
    # extend expiry
    obj.expires_at = max(obj.expires_at, datetime.utcnow()) + timedelta(days=days)
    db.commit()
    db.refresh(obj)
    db.close()
    return KeyOut(
        key=obj.key,
        created_at=obj.created_at,
        memo=obj.memo,
        expires_at=obj.expires_at
    )


# --- API key dependency with expiry check ---
def validate_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    db = SessionLocal()
    key = db.query(APIKey).filter(APIKey.key == x_api_key).first()
    db.close()
    if not key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    if key.expires_at < datetime.utcnow():
        raise HTTPException(status_code=401, detail="API Key expired")
    return x_api_key


# --- Health check ---
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow()}


# --- Protected data endpoint ---
@app.post("/data")
def receive_data(data: DataIn, api_key: str = Depends(validate_api_key)):
    return {"received": data.payload, "by": api_key}


# --- WebSocket & long‐task (unchanged) ---
class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, WebSocket] = {}

    async def connect(self, task_id: str, ws: WebSocket):
        await ws.accept()
        self.active[task_id] = ws

    def disconnect(self, task_id: str):
        self.active.pop(task_id, None)

    async def send_progress(self, task_id: str, msg: str):
        ws = self.active.get(task_id)
        if ws:
            await ws.send_json({"task_id": task_id, "progress": msg})

manager = ConnectionManager()

@app.post("/long-task", status_code=202)
def start_task(
    params: TaskStart,
    background: BackgroundTasks,
    api_key: str = Depends(validate_api_key)
):
    task_id = uuid.uuid4().hex
    background.add_task(long_task, task_id, params.work_units)
    return {"task_id": task_id}

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(ws: WebSocket, task_id: str):
    await manager.connect(task_id, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(task_id)

async def long_task(task_id: str, work_units: int):
    for i in range(1, work_units + 1):
        await asyncio.sleep(1)
        prog = f"{i}/{work_units}"
        await manager.send_progress(task_id, prog)
    await manager.send_progress(task_id, "done")
