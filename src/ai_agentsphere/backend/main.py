from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, and_, distinct, join, label, desc, asc, update, text
from database import get_db
from models import HubConversation, SatHistoricalChats, LnkUserAgentsConversation, SatUsersData, SatAgentsData, HubUser
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import hashlib
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {"status": "ok", "message": "Backend is running"}

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

    # 5. Usuarios Totales
    total_users = db.query(
        func.count(distinct(HubUser.kh_user))).filter(HubUser.ai_src_system == 31).scalar() or 0

    return {
        "total_conversations": total_convs,
        "feedback_percentage": round(feedback_pct, 2),
        "active_users": active_users,
        "top_user": top_user,
        "avg_response_time": "1.2s", # Este vendría de una métrica de performance
        "fallback_rate": "4.2%",
        "total_users": total_users
    }

@app.get("/api/dashboard/trend")
def get_trend_data(days: int = 7, db: Session = Depends(get_db)):
    # SQL nativo para generar los huecos y convertir zona horaria
    query = text("""
        WITH date_range AS (
            SELECT ( (CURRENT_TIMESTAMP AT TIME ZONE 'America/Mexico_City')::date - (i || ' day')::interval)::date as day
            FROM generate_series(0, :days - 1) i
        ),
        counts AS (
            SELECT 
                (ct_valid_from_dt AT TIME ZONE 'UTC' AT TIME ZONE 'America/Mexico_City')::date as day,
                COUNT(kh_user_agent_conversation) as total,
                BOOL_OR(ab_feedback = False) as has_negative_feedback
            FROM multiagent_rag_model.sat_compass_historical_chats
            WHERE ct_valid_from_dt >= (CURRENT_TIMESTAMP AT TIME ZONE 'America/Mexico_City')::date - (:days || ' day')::interval
            AND ax_message_type = 'bot_response'
            GROUP BY 1
        )
        SELECT 
            dr.day,
            COALESCE(c.total, 0) as mensajes,
            COALESCE(c.has_negative_feedback, False) as has_negative_feedback
        FROM date_range dr
        LEFT JOIN counts c ON dr.day = c.day
        ORDER BY dr.day ASC
    """)

    result_set = db.execute(query, {"days": days})

    result = []
    for row in result_set:
        fecha_str = row.day.strftime("%d %b")
        result.append({
            "fecha": fecha_str,
            "mensajes": row.mensajes,
            "has_negative_feedback": row.has_negative_feedback
        })
    
    return result

@app.get("/api/dashboard/combined")
def get_combined_dashboard(days: int = 7, db: Session = Depends(get_db)):
    # Reutilizamos la lógica de los otros dos endpoints
    stats = get_dashboard_stats(days, db)
    trend = get_trend_data(days, db)
    return {
        "stats": stats,
        "trend": trend
    }

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
            "timestamp": msg.ct_valid_from_dt.strftime("%d/%m/%Y %H:%M"),
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
                "comment": r.user_comment or "No comment",
                "snippet": r.snippet[:80] + "..." if r.snippet else ""
            } for r in results
        ]
    except Exception as e:
        # Esto imprimirá el error real en tu terminal negra (Uvicorn)
        print(f"DEBUG ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents/list")
def list_agents(db: Session = Depends(get_db)):
    agents = db.query(SatAgentsData).filter(SatAgentsData.ai_current_flag == 1).all()
    return [
        {
            "kh_agent": a.kh_agent.hex(),
            "name": a.ax_name,
            "description": a.ax_description
        } for a in agents
    ]

@app.get("/api/agents/detail/{kh_agent_hex}")
def get_agent_detail(kh_agent_hex: str, db: Session = Depends(get_db)):
    try:
        kh_bytes = bytes.fromhex(kh_agent_hex)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID de agente inválido")
    
    agent = db.query(SatAgentsData).filter(
        SatAgentsData.kh_agent == kh_bytes,
        SatAgentsData.ai_current_flag == 1
    ).first()
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agente no encontrado")
    
    return {
        "kh_agent": agent.kh_agent.hex(),
        "name": agent.ax_name,
        "description": agent.ax_description,
        "priming": agent.ax_priming,
        "agent_definition": agent.aj_agent_definition,
        "agent_examples": agent.aj_agent_examples,
        "is_supervisor": agent.ab_is_supervisor,
        "url": agent.ax_url
    }

@app.post("/api/agents/update")
async def update_agent_prompt(data: dict, db: Session = Depends(get_db)):
    try:
        # 1. Extraer datos del payload
        name = data.get('name')
        description = data.get('description')
        priming = data.get('priming')
        url = data.get('url')
        agent_def = data.get('agent_definition', {})
        agent_ex = data.get('agent_examples', {})
        is_sup = data.get('is_supervisor', False)

        # 2. Calcular el NUEVO Checksum
        def prepare_for_hash(val):
            if isinstance(val, (dict, list)):
                return json.dumps(val, sort_keys=True)
            return str(val) if val is not None else ""

        raw_string = f"{prepare_for_hash(name)}|{prepare_for_hash(description)}|{prepare_for_hash(url)}|{prepare_for_hash(priming)}|{prepare_for_hash(agent_def)}|{prepare_for_hash(agent_ex)}|{prepare_for_hash(is_sup)}"
        
        new_checksum_hex = hashlib.sha1(raw_string.encode('utf-8')).hexdigest()
        new_checksum_bytes = bytes.fromhex(new_checksum_hex)

        # 3. VALIDACIÓN: Comparar con el registro activo actual
        current_agent = db.query(SatAgentsData).filter(
            SatAgentsData.ai_current_flag == 1,
            SatAgentsData.ax_src_system_datastore == "gen-ai spin-compass"
        ).first()

        if current_agent and current_agent.ah_checksum == new_checksum_bytes:
            # Si los hashes son iguales, detenemos el proceso
            raise HTTPException(
                status_code=400, 
                detail="No se detectaron cambios en la configuración. La información es idéntica."
            )

        # 4. Si el checksum es diferente, procedemos con la actualización
        db.query(SatAgentsData).filter(
            SatAgentsData.ai_current_flag == 1,
            SatAgentsData.ax_src_system_datastore == "gen-ai spin-compass"
        ).update({"ai_current_flag": 0})

        new_version = SatAgentsData(
            kh_agent=bytes.fromhex(data.get('kh_agent')),
            ct_valid_from_dt=datetime.utcnow(),
            ct_ingest_dt=datetime.utcnow(),
            ah_checksum=new_checksum_bytes,
            ax_src_system_datastore="gen-ai spin-compass",
            ai_current_flag=1,
            ax_name=name,
            ax_description=description,
            aj_agent_definition=agent_def,
            ax_priming=priming,
            aj_agent_examples=agent_ex,
            ab_is_supervisor=is_sup,
            ax_url=url
        )
        
        db.add(new_version)
        db.commit()
        
        return {"status": "success", "checksum": new_checksum_hex}
        
    except HTTPException as he:
        raise he # Re-lanzamos el error de validación
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/history")
def get_agent_history(db: Session = Depends(get_db)):
    # Buscamos las 3 versiones anteriores más recientes
    history = db.query(SatAgentsData).filter(
        SatAgentsData.ai_current_flag == 0,
        SatAgentsData.ax_src_system_datastore == "gen-ai spin-compass"
    ).order_by(desc(SatAgentsData.ct_valid_from_dt)).limit(3).all()

    return [
        {
            "date": h.ct_valid_from_dt.strftime("%d/%m/%y %H:%M"),
            "description": h.ax_description,
            "priming": h.ax_priming
        } for h in history
    ]

@app.get("/api/agents/history/{kh_agent_hex}")
def get_specific_agent_history(kh_agent_hex: str, db: Session = Depends(get_db)):
    try:
        kh_bytes = bytes.fromhex(kh_agent_hex)
    except ValueError:
        raise HTTPException(status_code=400, detail="ID de agente inválido")

    history = db.query(SatAgentsData).filter(
        SatAgentsData.kh_agent == kh_bytes,
        SatAgentsData.ai_current_flag == 0
    ).order_by(desc(SatAgentsData.ct_valid_from_dt)).limit(3).all()

    return [
        {
            "date": h.ct_valid_from_dt.strftime("%d/%m/%y %H:%M"),
            "description": h.ax_description,
            "priming": h.ax_priming
        } for h in history
    ]