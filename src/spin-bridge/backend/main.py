from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, and_, distinct, join, label, desc, asc
from database import get_db
from models import HubConversation, SatHistoricalChats, LnkUserAgentsConversation, SatUsersData
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/dashboard/stats")
def get_dashboard_stats(days: int=7, db: Session = Depends(get_db)):
    #Calculamos la fecha de corte
    start_date = datetime.now() - timedelta(days=days)

    # 1. Total de conversaciones (Últimos n días)
    total_convs = db.query(func.count(SatHistoricalChats.kh_user_agent_conversation))\
    .filter(SatHistoricalChats.ax_message_type == "bot_response", SatHistoricalChats.ct_valid_from_dt >= start_date).scalar() or 0
    
    # 2. Feedback Positivo (Últimos n días)
    pos_feedback = db.query(func.count(SatHistoricalChats.kh_user_agent_conversation))\
                     .filter(SatHistoricalChats.ax_message_type == "bot_response", SatHistoricalChats.ct_valid_from_dt >= start_date,or_(
            SatHistoricalChats.ab_feedback == True,
            SatHistoricalChats.ab_feedback.is_(None)
        )).scalar() or 0
    
    # Cálculo de porcentaje
    feedback_pct = (pos_feedback / total_convs * 100) if total_convs > 0 else 0

    # 3. Usuarios Activos (Últimos n días)
    active_users = db.query(func.count(distinct(LnkUserAgentsConversation.kh_user)))\
    .join(SatHistoricalChats, LnkUserAgentsConversation.kh_user_agent_conversation == SatHistoricalChats.kh_user_agent_conversation)\
    .filter(SatHistoricalChats.ct_valid_from_dt >= start_date, SatHistoricalChats.ax_message_type == "user_query").scalar() or 0

    # 4. Top User (Últimos n días)
    top_user = db.query(SatUsersData.ax_display_nm).filter(SatUsersData.ai_current_flag == 1)\
    .join(LnkUserAgentsConversation, SatUsersData.kh_user == LnkUserAgentsConversation.kh_user)\
    .join(SatHistoricalChats, LnkUserAgentsConversation.kh_user_agent_conversation == SatHistoricalChats.kh_user_agent_conversation)\
    .filter(SatHistoricalChats.ct_valid_from_dt >= start_date, SatHistoricalChats.ax_message_type == "user_query")\
    .group_by(SatUsersData.ax_display_nm)\
    .order_by(func.count(SatHistoricalChats.kh_user_agent_conversation).desc())\
    .limit(1).scalar() or ""

    return {
        "total_conversations": total_convs,
        "feedback_percentage": round(feedback_pct, 2),
        "active_users": active_users,
        "top_user": top_user,
        "avg_response_time": "1.2s", # Este vendría de una métrica de performance
        "fallback_rate": "4.2%"
    }

@app.get("/api/dashboard/trend")
def get_trend_data(days: int = 7, db: Session = Depends(get_db)):
    # Usamos utcnow para consistencia con AlloyDB
    start_date = datetime.utcnow() - timedelta(days=days)
    
    trend_query = db.query(
        func.date(SatHistoricalChats.ct_valid_from_dt).label('fecha'), # Agregamos label
        func.count(SatHistoricalChats.kh_user_agent_conversation).label('total') # Agregamos label
    ).filter(
        SatHistoricalChats.ct_valid_from_dt >= start_date, 
        SatHistoricalChats.ax_message_type == "user_query"
    ).group_by(
        func.date(SatHistoricalChats.ct_valid_from_dt)
    ).order_by(
        func.date(SatHistoricalChats.ct_valid_from_dt)
    ).all()

    result = []
    for row in trend_query:
        # Al usar .label(), ahora sí puedes usar row.fecha
        fecha_obj = row.fecha 
        fecha_str = fecha_obj.strftime("%d %b") if hasattr(fecha_obj, 'strftime') else str(fecha_obj)
        
        result.append({
            "date": fecha_str,
            "conversations": row.total
        })
    
    return result

from sqlalchemy import desc, and_

@app.get("/api/conversations/negative/{kh_conv}")
def get_conversation_detail(kh_conv: str, db: Session = Depends(get_db)):
    try:
        # 1. Convertimos el string Hex (del Front) a Bytes (para la DB)
        # Esto es vital porque en Data Vault las llaves son binarias
        kh_bytes = bytes.fromhex(kh_conv)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID de conversación con formato inválido")
    
    # 2. Buscamos todos los registros (user y bot) de esa conversación
    messages = db.query(SatHistoricalChats.ax_message_type,
        SatHistoricalChats.ax_content,
        SatHistoricalChats.ct_valid_from_dt,
        SatHistoricalChats.ab_feedback,
        SatHistoricalChats.aj_attachments["comment"].label("comment"))\
        .filter(SatHistoricalChats.kh_user_agent_conversation == kh_bytes)\
        .order_by(asc(SatHistoricalChats.ct_valid_from_dt))\
        .all()

    if not messages:
        # Si no hay mensajes, devolvemos lista vacía en lugar de error
        return []

    # 3. Formateamos para las burbujas de chat
    return [
        {
            "type": msg.ax_message_type,      # 'user_query' o 'bot_response'
            "content": msg.ax_content,
            "timestamp": msg.ct_valid_from_dt.strftime("%H:%M"),
            "feedback": msg.ab_feedback,
            "is_bot": msg.ax_message_type == "bot_response",
            "comment": msg.comment
        } for msg in messages
    ]

@app.get("/api/conversations/negative")
def get_negative_conversations(
    days: int = Query(7), 
    email: str = Query(None), 
    db: Session = Depends(get_db)
):
    try:
        # Usamos utcnow para evitar desfases de horario con la DB
        start_date = datetime.utcnow() - timedelta(days=days)

        query = db.query(
            SatHistoricalChats.kh_user_agent_conversation.label("id"),
            SatHistoricalChats.ct_valid_from_dt.label("date"),
            SatHistoricalChats.ax_content.label("snippet"),
            SatUsersData.ax_email.label("email"),
            SatUsersData.ax_display_nm.label("user_name"),
            SatHistoricalChats.aj_attachments["comment"].label("user_comment")
        ).join(
            LnkUserAgentsConversation, 
            SatHistoricalChats.kh_user_agent_conversation == LnkUserAgentsConversation.kh_user_agent_conversation
        ).join(
            SatUsersData,
            LnkUserAgentsConversation.kh_user == SatUsersData.kh_user
        ).filter(
            SatHistoricalChats.ax_message_type == "bot_response",
            SatHistoricalChats.ct_valid_from_dt >= start_date,
            SatHistoricalChats.ab_feedback == False, 
            SatUsersData.ai_current_flag == 1
        )

        if email and email.strip():
            query = query.filter(SatUsersData.ax_email.ilike(f"%{email}%"))

        results = query.order_by(desc(SatHistoricalChats.ct_valid_from_dt)).all()

        return [
            {
                "kh_conversation": r.id.hex(),
                "date": r.date.strftime("%d/%m/%y %H:%M"),
                "user": r.user_name or r.email,
                "email": r.email,
                "comment": r.user_comment or "Sin comentario",
                "snippet": r.snippet[:80] + "..." if r.snippet else ""
            } for r in results
        ]
    except Exception as e:
        # Esto imprimirá el error real en tu terminal negra (Uvicorn)
        print(f"DEBUG ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))