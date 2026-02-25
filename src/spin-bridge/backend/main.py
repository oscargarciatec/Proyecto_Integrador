from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from .database import get_db
from .models import HubConversation, SatHistoricalChats
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/dashboard/stats")
def get_dashboard_stats(db: Session = Depends(get_db)):
    # 1. Total de conversaciones (de Hubs)
    total_convs = db.query(func.count(HubConversation.kh_conversation)).scalar()
    
    # 2. Feedback Positivo (de Satellites)
    pos_feedback = db.query(func.count(SatHistoricalChats.kh_user_agent_conversation))\
                     .filter(SatHistoricalChats.ab_feedback == True).scalar()
    
    # Cálculo de porcentaje
    feedback_pct = (pos_feedback / total_convs * 100) if total_convs > 0 else 0
    
    return {
        "total_conversations": total_convs,
        "feedback_percentage": round(feedback_pct, 1),
        "avg_response_time": "1.2s", # Este vendría de una métrica de performance
        "fallback_rate": "4.2%"
    }