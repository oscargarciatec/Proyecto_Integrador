"""
Estrategias para determinar cuándo crear una nueva conversación.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

class ConversationStrategy:
    """Estrategia base para determinar business key de conversación."""
    
    def get_business_key(self, user_id: str, channel_id: str, 
                        thread_ts: Optional[str]) -> str:
        raise NotImplementedError


class ThreadBasedStrategy(ConversationStrategy):
    """
    Estrategia actual: Basada en thread de Slack.
    - Si hay thread_ts → cada thread es una conversación
    - Si no hay thread_ts → todo el canal/DM es una conversación
    """
    
    def get_business_key(self, user_id: str, channel_id: str, 
                        thread_ts: Optional[str]) -> str:
        return f"{channel_id}_{thread_ts}" if thread_ts else channel_id


class SessionBasedStrategy(ConversationStrategy):
    """
    Estrategia basada en sesiones con timeout configurable.
    
    - Con thread → usa thread_ts (comportamiento Slack)
    - Sin thread → crea nueva conversación si pasan X horas de inactividad
    
    Útil para DMs donde no hay threads pero quieres separar por sesiones.
    """
    
    def __init__(self, session_timeout_hours: float = 2.0, engine=None, db_lock: asyncio.Lock = None, exec_async_fn=None):
        """
        Args:
            session_timeout_hours: Horas de inactividad para nueva conversación
            engine: AlloyDBEngine para consultar BD (legacy, preferir exec_async_fn)
            db_lock: Lock compartido para serializar operaciones de BD
            exec_async_fn: Función async para ejecutar queries (preferido)
        """
        self.session_timeout_hours = session_timeout_hours
        self.engine = engine
        self._db_lock = db_lock
        self._exec_async_fn = exec_async_fn
        self._cache = {}
    
    def get_business_key(self, user_id: str, channel_id: str, 
                        thread_ts: Optional[str]) -> str:
        """Versión sync - solo para estrategias que no hacen I/O."""
        if thread_ts:
            return f"{channel_id}_{thread_ts}"
        # Sin engine, crear nueva sesión
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{user_id}_{channel_id}_{session_id}"
    
    async def get_business_key_async(self, user_id: str, channel_id: str, 
                                     thread_ts: Optional[str]) -> str:
        """Versión async - usa queries async a la BD."""
        if thread_ts:
            return f"{channel_id}_{thread_ts}"
        
        now = datetime.now()
        cache_key = (user_id, channel_id)
        
        # Verificar cache
        cached = self._cache.get(cache_key)
        if cached and (now - cached['timestamp']).total_seconds() < 2:
            return cached['bk']
        
        # Buscar última conversación
        last_conversation = await self._get_last_conversation_async(user_id, channel_id)
        
        import logging
        logger = logging.getLogger(__name__)
        
        if last_conversation:
            last_activity = last_conversation['timestamp']
            last_bk = last_conversation['business_key']
            time_since_last = now - last_activity
            
            logger.info(f"SessionBasedStrategy: Time since last: {time_since_last.total_seconds():.1f}s (timeout: {self.session_timeout_hours * 3600:.1f}s)")
            
            if time_since_last > timedelta(hours=self.session_timeout_hours):
                session_id = now.strftime("%Y%m%d_%H%M%S")
                conversation_bk = f"{user_id}_{channel_id}_{session_id}"
                logger.info(f"SessionBasedStrategy: TIMEOUT - New conversation: {conversation_bk}")
            else:
                conversation_bk = last_bk
                logger.info(f"SessionBasedStrategy: Reusing conversation: {conversation_bk}")
        else:
            session_id = now.strftime("%Y%m%d_%H%M%S")
            conversation_bk = f"{user_id}_{channel_id}_{session_id}"
            logger.info(f"SessionBasedStrategy: First conversation: {conversation_bk}")
        
        self._cache[cache_key] = {'bk': conversation_bk, 'timestamp': now}
        return conversation_bk
    
    async def _get_last_conversation_async(self, user_id: str, channel_id: str) -> Optional[dict]:
        """Obtiene la última conversación de forma async."""
        if not self.engine:
            return None
        
        try:
            from sqlalchemy import text
            import logging
            logger = logging.getLogger(__name__)
            
            pattern = f"{user_id}_{channel_id}_%"
            logger.info(f"SessionBasedStrategy: Querying DB for pattern: {pattern}")
            
            query = text("""
                SELECT 
                    c.ax_conversation,
                    COALESCE(MAX(s.ct_ingest_dt), c.ct_ingest_dt) as last_activity
                FROM multiagent_rag_model.hub_conversation c
                LEFT JOIN multiagent_rag_model.lnk_users_agents_conversation l 
                    ON c.kh_conversation = l.kh_conversation
                LEFT JOIN multiagent_rag_model.sat_compass_current_chat s 
                    ON l.kh_user_agent_conversation = s.kh_user_agent_conversation
                WHERE c.ax_conversation LIKE :pattern
                GROUP BY c.ax_conversation, c.ct_ingest_dt
                ORDER BY last_activity DESC
                LIMIT 1
            """)
            
            # Usar exec_async_fn si está disponible (ya tiene el lock)
            if self._exec_async_fn:
                rows = await self._exec_async_fn(query, {'pattern': pattern}, fetch=True)
                row = rows[0] if rows else None
            else:
                # Fallback: conexión directa con lock
                async def _run_query():
                    async with self.engine._pool.connect() as conn:
                        result = await conn.execute(query, {'pattern': pattern})
                        return result.fetchone()

                if self._db_lock:
                    async with self._db_lock:
                        row = await _run_query()
                else:
                    row = await _run_query()
            
            if row:
                logger.info(f"SessionBasedStrategy: Found: {row[0]}, timestamp: {row[1]}")
                return {'business_key': row[0], 'timestamp': row[1]}
            
            logger.info(f"SessionBasedStrategy: No previous conversation for {user_id}")
            return None
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"SessionBasedStrategy: Error querying DB: {e}", exc_info=True)
            return None


class DailyBasedStrategy(ConversationStrategy):
    """
    Estrategia diaria: Nueva conversación cada día.
    
    - Con thread → usa thread_ts
    - Sin thread → crea conversación por día (YYYY-MM-DD)
    """
    
    def get_business_key(self, user_id: str, channel_id: str, 
                        thread_ts: Optional[str]) -> str:
        if thread_ts:
            return f"{channel_id}_{thread_ts}"
        
        # Sin thread: una conversación por día
        today = datetime.now().strftime("%Y-%m-%d")
        return f"{channel_id}_{today}"


class AlwaysNewStrategy(ConversationStrategy):
    """
    Estrategia siempre nueva: Cada mensaje inicia nueva conversación.
    Útil para analytics donde quieres trackear cada interacción separadamente.
    """
    
    def get_business_key(self, user_id: str, channel_id: str, 
                        thread_ts: Optional[str]) -> str:
        if thread_ts:
            return f"{channel_id}_{thread_ts}"
        
        # Sin thread: usar timestamp actual como identificador único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{channel_id}_{user_id}_{timestamp}"


class UserSessionStrategy(ConversationStrategy):
    """
    Estrategia por usuario: Cada usuario tiene su propia sesión continua.
    
    - Con thread → usa thread_ts
    - Sin thread → usa user_id como parte del business key
    
    Útil para mantener contexto largo entre usuario y bot.
    """
    
    def __init__(self, session_timeout_hours: int = 24):
        self.session_timeout_hours = session_timeout_hours
        self._user_sessions = {}  # {user_id: (session_id, last_activity)}
    
    def get_business_key(self, user_id: str, channel_id: str, 
                        thread_ts: Optional[str]) -> str:
        if thread_ts:
            return f"{channel_id}_{thread_ts}"
        
        # Sin thread: sesión por usuario
        now = datetime.now()
        
        if user_id in self._user_sessions:
            session_id, last_activity = self._user_sessions[user_id]
            time_since_last = now - last_activity
            
            if time_since_last > timedelta(hours=self.session_timeout_hours):
                # Nueva sesión para este usuario
                session_id = now.strftime("%Y%m%d_%H%M%S")
                self._user_sessions[user_id] = (session_id, now)
            else:
                # Actualizar última actividad
                self._user_sessions[user_id] = (session_id, now)
        else:
            # Primera sesión del usuario
            session_id = now.strftime("%Y%m%d_%H%M%S")
            self._user_sessions[user_id] = (session_id, now)
        
        return f"{channel_id}_{user_id}_{session_id}"


# Factory para crear estrategias
def create_strategy(strategy_type: str = "thread", **kwargs) -> ConversationStrategy:
    """
    Factory para crear estrategias de conversación.
    
    Args:
        strategy_type: Tipo de estrategia
            - "thread": Basada en threads de Slack (default)
            - "session": Sesiones con timeout
            - "daily": Una conversación por día
            - "always_new": Cada mensaje es nueva conversación
            - "user_session": Sesión por usuario
        **kwargs: Parámetros adicionales para la estrategia
    
    Returns:
        Instancia de ConversationStrategy
    """
    strategies = {
        "thread": ThreadBasedStrategy,
        "session": SessionBasedStrategy,
        "daily": DailyBasedStrategy,
        "always_new": AlwaysNewStrategy,
        "user_session": UserSessionStrategy
    }
    
    strategy_class = strategies.get(strategy_type)
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {strategy_type}. Available: {list(strategies.keys())}")
    
    return strategy_class(**kwargs)
