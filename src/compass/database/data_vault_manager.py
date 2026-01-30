# database/data_vault_manager.py
"""
Adapter para trabajar con el esquema Data Vault 2.0 existente en AlloyDB.
Reemplaza el PersistenceManager para usar las tablas reales:
- HUBs: hub_users, hub_agents, hub_conversation, hub_knowledge
- LINKs: lnk_users_agents_conversation, lnk_agents_interactions, lnk_agents_knowledge, lnk_context
- SATs: sat_compass_users_data, sat_compass_current_chat, sat_compass_historical_chats, sat_knowledge, etc.
"""

import hashlib
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from sqlalchemy import text
from database.conversation_strategies import create_strategy

logger = logging.getLogger(__name__)


class DataVaultManager:
    """
    Maneja la persistencia usando el esquema Data Vault 2.0 real.
    """

    def __init__(
        self,
        engine,
        agent_id: str = "Compass",
        agent_definition: Optional[Dict[str, Any]] = None,
        agent_name: str = "Compass-Slack",
        agent_description: str = "Agente RAG sobre documentos de políticas de la empresa.",
        conversation_strategy: str = "session",
        strategy_kwargs: Optional[Dict] = None,
        skip_agent_init: bool = False,
    ):
        """
        Args:
            engine: AlloyDBEngine instance
            agent_id: Identificador del agente (para hub_agents)
            conversation_strategy: Tipo de estrategia para conversaciones
                - "thread" (default antiguo): Basada en threads de Slack
                - "session" (recomendado): Sesiones con timeout
                - "daily": Una conversación por día
                - "always_new": Cada mensaje es nueva conversación
                - "user_session": Sesión continua por usuario
            strategy_kwargs: Parámetros para la estrategia (ej: session_timeout_hours=2)
            skip_agent_init: Si True, no inicializa el agente (para evitar deadlock en contexto async)
        """
        self.engine = engine
        self.schema = "multiagent_rag_model"
        self.agent_id = agent_id
        self.agent_definition = agent_definition or {}
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.src_system = "gen-ai spin-compass"
        self.src_system_id = 31

        # Estrategia de conversación
        strategy_kwargs = strategy_kwargs or {}
        # Pasar engine a la estrategia
        if conversation_strategy == "session":
            strategy_kwargs["engine"] = self.engine
            strategy_kwargs["exec_async_fn"] = self._exec_async

        self.conversation_strategy = create_strategy(
            conversation_strategy, **strategy_kwargs
        )

        # NOTA: La inicialización del agente debe hacerse con ensure_agent_exists_async()
        # después de crear el manager en un contexto async

    # ==================== HELPERS ====================

    async def _exec_async(self, statement, params=None, fetch=False, commit=False):
        """Helper async para ejecutar SQLAlchemy text statements."""
        async with self.engine._pool.connect() as conn:
            result = await conn.execute(statement, params or {})
            rows = result.fetchall() if fetch else None
            if commit:
                await conn.commit()
            return rows

    async def _exec_on_conn_async(self, conn, statement, params=None, fetch=False):
        """Ejecuta un statement usando una conexión ya abierta (para transacciones)."""
        result = await conn.execute(statement, params or {})
        return result.fetchall() if fetch else None

    # ==================== HASH KEYS ====================

    def _generate_hash_key(self, business_key: str) -> bytes:
        """
        Genera hash key (kh_*) para Data Vault.
        Usa SHA-1 con formato: sha1(id + '|' + src_system)
        """
        composite_key = f"{business_key}|{self.src_system}"
        return hashlib.sha1(composite_key.encode("utf-8")).digest()

    @staticmethod
    def _generate_checksum(data: Dict) -> bytes:
        """
        Genera checksum (ah_*) para detectar cambios en SATs como bytea (SHA-1).
        """
        # Ordenar dict para hash consistente
        data_str = json.dumps(
            data, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.sha1(data_str.encode("utf-8")).digest()

    @staticmethod
    def _clean_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Elimina llaves con valores vacíos para evitar ruido en el contexto."""
        return {
            k: v
            for k, v in data.items()
            if v not in (None, "", [], {}, ())
        }

    def _build_user_context_payload(self, slack_user: Dict[str, Any]) -> Dict[str, Any]:
        """Construye un payload rico en contexto para ax_user_context."""
        profile = slack_user.get("profile") or {}
        context = {
            "slack_id": slack_user.get("id") or slack_user.get("user"),
            "team_id": slack_user.get("team_id"),
            "email": slack_user.get("email") or profile.get("email"),
            "display_name": slack_user.get("display_name")
            or profile.get("display_name")
            or slack_user.get("real_name"),
            "real_name": slack_user.get("real_name") or profile.get("real_name"),
            "username": slack_user.get("username") or slack_user.get("name"),
            "title": profile.get("title"),
            "phone": profile.get("phone"),
            "image_512": profile.get("image_512"),
            "image_192": profile.get("image_192"),
            "timezone": slack_user.get("tz"),
            "timezone_label": slack_user.get("tz_label"),
            "locale": slack_user.get("locale") or profile.get("locale"),
            "profile_fields": profile.get("fields"),
        }

        sanitized = self._clean_dict(context)
        if profile:
            sanitized["profile"] = profile
        return sanitized

    # ==================== AGENT ====================

    async def ensure_agent_exists_async(self):
        """Crea el agente en hub_agents si no existe (versión async)."""
        try:
            kh_agent = self._generate_hash_key(self.agent_id)

            ensure_hub_query = text(f"""
                SELECT 1 FROM {self.schema}.hub_agents
                WHERE kh_agent = :kh_agent
            """)

            async with self.engine._pool.connect() as conn:
                result = await conn.execute(ensure_hub_query, {"kh_agent": kh_agent})
                hub_row = result.fetchone()
                if not hub_row:
                    insert_hub = text(f"""
                        INSERT INTO {self.schema}.hub_agents
                        (kh_agent, ax_agent, ct_ingest_dt, ax_src_system_datastore, ai_src_system)
                        VALUES (:kh_agent, :ax_agent, :ct_ingest_dt, :ax_src_system, :ai_src_system)
                    """)

                    await conn.execute(
                        insert_hub,
                        {
                            "kh_agent": kh_agent,
                            "ax_agent": self.agent_id,
                            "ct_ingest_dt": datetime.now(),
                            "ax_src_system": self.src_system,
                            "ai_src_system": self.src_system_id,
                        },
                    )
                    await conn.commit()

                agent_data = {
                    "name": self.agent_name,
                    "description": self.agent_description,
                    "definition": self.agent_definition,
                }
                new_checksum = self._generate_checksum(agent_data)

                current_sat_query = text(f"""
                    SELECT ah_checksum
                    FROM {self.schema}.sat_agents_data
                    WHERE kh_agent = :kh_agent AND ai_current_flag = 1
                    ORDER BY ct_valid_from_dt DESC
                    LIMIT 1
                """)

                current_sat_result = await conn.execute(
                    current_sat_query, {"kh_agent": kh_agent}
                )
                current_sat_row = current_sat_result.fetchone()

                current_checksum = None
                if current_sat_row and current_sat_row[0] is not None:
                    current_checksum = bytes(current_sat_row[0])

                if current_checksum is not None and current_checksum == new_checksum:
                    logger.info(f"Agent {self.agent_id} already exists")
                    return

                if current_sat_row:
                    update_old = text(f"""
                        UPDATE {self.schema}.sat_agents_data
                        SET ai_current_flag = 0
                        WHERE kh_agent = :kh_agent AND ai_current_flag = 1
                    """)
                    await conn.execute(update_old, {"kh_agent": kh_agent})
                    await conn.commit()

                insert_sat = text(f"""
                    INSERT INTO {self.schema}.sat_agents_data
                    (kh_agent, ct_valid_from_dt, ai_current_flag, ct_ingest_dt,
                     ax_src_system_datastore, ah_checksum, ax_name, ax_description,
                     ax_url, aj_agent_definition, aj_priming, aj_agent_examples, ab_is_supervisor)
                    VALUES (:kh_agent, :ct_valid_from_dt, :ai_current_flag, :ct_ingest_dt,
                            :ax_src_system, :ah_checksum, :ax_name, :ax_description,
                            :ax_url, :aj_agent_definition, :aj_priming, :aj_agent_examples, :ab_is_supervisor)
                """)

                await conn.execute(
                    insert_sat,
                    {
                        "kh_agent": kh_agent,
                        "ct_valid_from_dt": datetime.now(),
                        "ai_current_flag": 1,
                        "ct_ingest_dt": datetime.now(),
                        "ax_src_system": self.src_system,
                        "ah_checksum": new_checksum,
                        "ax_name": self.agent_name,
                        "ax_description": self.agent_description,
                        "ax_url": None,
                        "aj_agent_definition": json.dumps(
                            self.agent_definition, ensure_ascii=False
                        ),
                        "aj_priming": json.dumps({}),
                        "aj_agent_examples": json.dumps({}),
                        "ab_is_supervisor": False,
                    },
                )
                await conn.commit()

            logger.info(f"Agent {self.agent_id} created successfully")

        except Exception as e:
            logger.error(f"Error ensuring agent exists: {e}")

    # ==================== USUARIOS ====================

    async def upsert_user_async(self, slack_user: Dict[str, Any]) -> tuple[bytes, str]:
        """
        Inserta o actualiza un usuario en hub_users y sat_compass_users_data (async).
        
        WARNING: Logic duplicated in save_interaction_atomic_async.
        If you update SQL here, please update it there too.
        """
        profile = slack_user.get("profile") or {}
        email = slack_user.get("email") or profile.get("email")
        slack_id = slack_user.get("id") or slack_user.get("user")

        if email:
            user_key = email
        elif slack_id:
            user_key = slack_id
            logger.warning(f"User {slack_id} has no email, using Slack ID as key")
        else:
            logger.warning("No user_id or email found in slack_user data")
            return None, None

        try:
            kh_user = self._generate_hash_key(user_key)

            # 1. Upsert en HUB (idempotente)
            upsert_hub = text(f"""
                INSERT INTO {self.schema}.hub_users 
                (kh_user, ax_user, ct_ingest_dt, ax_src_system_datastore, ai_src_system)
                VALUES (:kh_user, :ax_user, :ct_ingest_dt, :ax_src_system, :ai_src_system)
                ON CONFLICT (kh_user) DO NOTHING
            """)

            await self._exec_async(
                upsert_hub,
                {
                    "kh_user": kh_user,
                    "ax_user": user_key,
                    "ct_ingest_dt": datetime.now(),
                    "ax_src_system": self.src_system,
                    "ai_src_system": self.src_system_id,
                },
                commit=True,
            )

            # 2. Verificar si hay cambios en los datos (checksum)
            display_name = (
                slack_user.get("display_name")
                or slack_user.get("real_name")
                or slack_user.get("username")
                or slack_user.get("name")
                or profile.get("display_name")
                or profile.get("real_name")
            )
            if display_name:
                slack_user.setdefault("display_name", display_name)
            if email and not slack_user.get("email"):
                slack_user["email"] = email

            job_title = (
                slack_user.get("title")
                or profile.get("title")
            )
            user_context = self._build_user_context_payload(slack_user)
            user_data = {
                "display_name": display_name,
                "email": email,
                "job_title": job_title,
                "slack_id": slack_id,
                "user_context": user_context,
            }
            new_checksum = self._generate_checksum(user_data)

            check_query = text(f"""
                SELECT ah_checksum FROM {self.schema}.sat_compass_users_data 
                WHERE kh_user = :kh_user AND ai_current_flag = 1
            """)
            rows = await self._exec_async(check_query, {"kh_user": kh_user}, fetch=True)
            row = rows[0] if rows else None
            current_checksum = bytes(row[0]) if row else None

            if not row or current_checksum != new_checksum:
                if row:
                    update_old = text(f"""
                        UPDATE {self.schema}.sat_compass_users_data 
                        SET ai_current_flag = 0
                        WHERE kh_user = :kh_user AND ai_current_flag = 1
                    """)
                    await self._exec_async(update_old, {"kh_user": kh_user}, commit=True)

                insert_sat = text(f"""
                    INSERT INTO {self.schema}.sat_compass_users_data 
                    (kh_user, ct_valid_from_dt, ai_current_flag, ct_ingest_dt, 
                     ax_src_system_datastore, ah_checksum, ax_display_nm, 
                     ax_user_context, ax_job_title, ax_email)
                    VALUES (:kh_user, :ct_valid_from_dt, :ai_current_flag, :ct_ingest_dt,
                            :ax_src_system, :ah_checksum, :ax_display_nm,
                            :ax_user_context, :ax_job_title, :ax_email)
                """)

                await self._exec_async(
                    insert_sat,
                    {
                        "kh_user": kh_user,
                        "ct_valid_from_dt": datetime.now(),
                        "ai_current_flag": 1,
                        "ct_ingest_dt": datetime.now(),
                        "ax_src_system": self.src_system,
                        "ah_checksum": new_checksum,
                        "ax_display_nm": user_data["display_name"],
                        "ax_user_context": json.dumps(user_context, ensure_ascii=False),
                        "ax_job_title": user_data["job_title"],
                        "ax_email": user_data["email"],
                    },
                    commit=True,
                )

            logger.info(f"User {user_key} upserted successfully")
            return kh_user, user_key

        except Exception as e:
            logger.error(f"Error upserting user: {e}")
            return self._generate_hash_key(user_key), user_key

    async def save_interaction_atomic_async(
        self,
        slack_user: Dict[str, Any],
        channel_id: str,
        thread_ts: str | None,
        query: str,
        output: str,
    ) -> Dict[str, Any]:
        """Guarda una interacción completa en una sola transacción."""
        profile = slack_user.get("profile") or {}
        email = slack_user.get("email") or profile.get("email")
        slack_id = slack_user.get("id") or slack_user.get("user")

        if email:
            user_key = email
        elif slack_id:
            user_key = slack_id
            logger.warning(f"User {slack_id} has no email, using Slack ID as key")
        else:
            logger.warning("No user_id or email found in slack_user data")
            return {}

        try:
            conversation_bk = await self._get_conversation_bk_async(
                user_key=user_key,
                channel_id=channel_id,
                thread_ts=thread_ts,
            )
            kh_user = self._generate_hash_key(user_key)
            kh_agent = self._generate_hash_key(self.agent_id)
            kh_conversation = self._generate_hash_key(conversation_bk)

            link_bk = f"{user_key}_{self.agent_id}_{conversation_bk}"
            kh_link = self._generate_hash_key(link_bk)

            async with self.engine._pool.connect() as conn:
                async with conn.begin():
                    upsert_hub_user = text(f"""
                        INSERT INTO {self.schema}.hub_users
                        (kh_user, ax_user, ct_ingest_dt, ax_src_system_datastore, ai_src_system)
                        VALUES (:kh_user, :ax_user, :ct_ingest_dt, :ax_src_system, :ai_src_system)
                        ON CONFLICT (kh_user) DO NOTHING
                    """)
                    await self._exec_on_conn_async(
                        conn,
                        upsert_hub_user,
                        {
                            "kh_user": kh_user,
                            "ax_user": user_key,
                            "ct_ingest_dt": datetime.now(),
                            "ax_src_system": self.src_system,
                            "ai_src_system": self.src_system_id,
                        },
                    )

                    display_name = (
                        slack_user.get("display_name")
                        or slack_user.get("real_name")
                        or slack_user.get("username")
                        or slack_user.get("name")
                        or profile.get("display_name")
                        or profile.get("real_name")
                    )
                    if display_name:
                        slack_user.setdefault("display_name", display_name)
                    if email and not slack_user.get("email"):
                        slack_user["email"] = email

                    job_title = slack_user.get("title") or profile.get("title")
                    user_context = self._build_user_context_payload(slack_user)
                    user_data = {
                        "display_name": display_name,
                        "email": email,
                        "job_title": job_title,
                        "slack_id": slack_id,
                        "user_context": user_context,
                    }
                    new_checksum = self._generate_checksum(user_data)

                    check_user_sat = text(f"""
                        SELECT ah_checksum FROM {self.schema}.sat_compass_users_data
                        WHERE kh_user = :kh_user AND ai_current_flag = 1
                        ORDER BY ct_valid_from_dt DESC
                        LIMIT 1
                    """)
                    rows = await self._exec_on_conn_async(
                        conn, check_user_sat, {"kh_user": kh_user}, fetch=True
                    )
                    row = rows[0] if rows else None
                    current_checksum = bytes(row[0]) if row and row[0] is not None else None

                    if current_checksum != new_checksum:
                        update_old_user_sat = text(f"""
                            UPDATE {self.schema}.sat_compass_users_data
                            SET ai_current_flag = 0
                            WHERE kh_user = :kh_user AND ai_current_flag = 1
                        """)
                        await self._exec_on_conn_async(
                            conn, update_old_user_sat, {"kh_user": kh_user}
                        )

                        insert_user_sat = text(f"""
                            INSERT INTO {self.schema}.sat_compass_users_data
                            (kh_user, ct_valid_from_dt, ai_current_flag, ct_ingest_dt,
                             ax_src_system_datastore, ah_checksum, ax_display_nm,
                             ax_user_context, ax_job_title, ax_email)
                            VALUES (:kh_user, :ct_valid_from_dt, :ai_current_flag, :ct_ingest_dt,
                                    :ax_src_system, :ah_checksum, :ax_display_nm,
                                    :ax_user_context, :ax_job_title, :ax_email)
                        """)
                        await self._exec_on_conn_async(
                            conn,
                            insert_user_sat,
                            {
                                "kh_user": kh_user,
                                "ct_valid_from_dt": datetime.now(),
                                "ai_current_flag": 1,
                                "ct_ingest_dt": datetime.now(),
                                "ax_src_system": self.src_system,
                                "ah_checksum": new_checksum,
                                "ax_display_nm": user_data["display_name"],
                                "ax_user_context": json.dumps(user_context, ensure_ascii=False),
                                "ax_job_title": user_data["job_title"],
                                "ax_email": user_data["email"],
                            },
                        )

                    upsert_hub_conversation = text(f"""
                        INSERT INTO {self.schema}.hub_conversation
                        (kh_conversation, ax_conversation, ct_ingest_dt, ax_src_system_datastore, ai_src_system)
                        VALUES (:kh_conversation, :ax_conversation, :ct_ingest_dt, :ax_src_system, :ai_src_system)
                        ON CONFLICT (kh_conversation) DO NOTHING
                    """)
                    await self._exec_on_conn_async(
                        conn,
                        upsert_hub_conversation,
                        {
                            "kh_conversation": kh_conversation,
                            "ax_conversation": conversation_bk,
                            "ct_ingest_dt": datetime.now(),
                            "ax_src_system": self.src_system,
                            "ai_src_system": self.src_system_id,
                        },
                    )

                    upsert_link = text(f"""
                        INSERT INTO {self.schema}.lnk_users_agents_conversation
                        (kh_user_agent_conversation, kh_user, kh_agent, kh_conversation,
                         ct_ingest_dt, ax_src_system_datastore, ai_src_system)
                        VALUES (:kh_link, :kh_user, :kh_agent, :kh_conversation,
                                :ct_ingest_dt, :ax_src_system, :ai_src_system)
                        ON CONFLICT (kh_user_agent_conversation) DO NOTHING
                    """)
                    await self._exec_on_conn_async(
                        conn,
                        upsert_link,
                        {
                            "kh_link": kh_link,
                            "kh_user": kh_user,
                            "kh_agent": kh_agent,
                            "kh_conversation": kh_conversation,
                            "ct_ingest_dt": datetime.now(),
                            "ax_src_system": self.src_system,
                            "ai_src_system": self.src_system_id,
                        },
                    )

                    async def insert_chat(message_type: str, content: str, feedback: bool | None):
                        chat_data = {"content": content, "type": message_type}
                        checksum = self._generate_checksum(chat_data)
                        now = datetime.now()
                        params = {
                            "kh_link": kh_link,
                            "ct_valid_from_dt": now,
                            "ct_ingest_dt": now,
                            "ax_src_system": self.src_system,
                            "ah_checksum": checksum,
                            "ax_message_type": message_type,
                            "ab_feedback": feedback,
                            "ax_content": content,
                            "aj_attachments": json.dumps({}),
                        }
                        insert_current = text(f"""
                            INSERT INTO {self.schema}.sat_compass_current_chat
                            (kh_user_agent_conversation, ct_valid_from_dt, ct_ingest_dt,
                             ax_src_system_datastore, ah_checksum, ax_message_type,
                             ab_feedback, ax_content, aj_attachments)
                            VALUES (:kh_link, :ct_valid_from_dt, :ct_ingest_dt,
                                    :ax_src_system, :ah_checksum, :ax_message_type,
                                    :ab_feedback, :ax_content, :aj_attachments)
                        """)
                        insert_historical = text(f"""
                            INSERT INTO {self.schema}.sat_compass_historical_chats
                            (kh_user_agent_conversation, ct_valid_from_dt, ct_ingest_dt,
                             ax_src_system_datastore, ah_checksum, ax_message_type,
                             ab_feedback, ax_content, aj_attachments)
                            VALUES (:kh_link, :ct_valid_from_dt, :ct_ingest_dt,
                                    :ax_src_system, :ah_checksum, :ax_message_type,
                                    :ab_feedback, :ax_content, :aj_attachments)
                        """)
                        await self._exec_on_conn_async(conn, insert_current, params)
                        await self._exec_on_conn_async(conn, insert_historical, params)

                    await insert_chat("user_query", query, None)
                    await insert_chat("bot_response", output, None)

            return {
                "kh_user": kh_user,
                "user_key": user_key,
                "kh_conversation": kh_conversation,
                "query_saved": True,
                "response_saved": True,
            }

        except Exception as e:
            logger.exception(f"Error saving interaction atomically: {e}")
            return {}

    # ==================== CONVERSACIONES ====================

    async def _ensure_link_exists_async(
        self, kh_link: bytes, kh_user: bytes, kh_agent: bytes, kh_conversation: bytes
    ) -> None:
        """Verifica si el link existe, si no lo crea."""
        check_link = text(f"""
            SELECT 1 FROM {self.schema}.lnk_users_agents_conversation 
            WHERE kh_user_agent_conversation = :kh_link
        """)
        link_exists = await self._exec_async(check_link, {"kh_link": kh_link}, fetch=True)
        
        if not link_exists:
            insert_link = text(f"""
                INSERT INTO {self.schema}.lnk_users_agents_conversation 
                (kh_user_agent_conversation, kh_user, kh_agent, kh_conversation, 
                 ct_ingest_dt, ax_src_system_datastore, ai_src_system)
                VALUES (:kh_link, :kh_user, :kh_agent, :kh_conversation,
                        :ct_ingest_dt, :ax_src_system, :ai_src_system)
            """)
            await self._exec_async(
                insert_link,
                {
                    "kh_link": kh_link,
                    "kh_user": kh_user,
                    "kh_agent": kh_agent,
                    "kh_conversation": kh_conversation,
                    "ct_ingest_dt": datetime.now(),
                    "ax_src_system": self.src_system,
                    "ai_src_system": self.src_system_id,
                },
                commit=True,
            )
            logger.info("Link created for existing conversation")

    async def _get_conversation_bk_async(
        self, user_key: str, channel_id: str, thread_ts: str = None
    ) -> str:
        """
        Helper async para obtener el business key de una conversación.
        Si la estrategia es SessionBasedStrategy, usa versión async.
        """
        strategy = self.conversation_strategy
        
        # Si es SessionBasedStrategy, usar versión async
        if hasattr(strategy, 'get_business_key_async'):
            return await strategy.get_business_key_async(
                user_id=user_key,
                channel_id=channel_id,
                thread_ts=thread_ts,
            )
        
        # Para otras estrategias, usar versión sync (no hacen I/O)
        return strategy.get_business_key(
            user_id=user_key,
            channel_id=channel_id,
            thread_ts=thread_ts,
        )

    async def get_or_create_conversation_async(
        self, user_key: str, channel_id: str, thread_ts: str = None
    ) -> bytes:
        """Obtiene o crea una conversación (async)."""
        
        # WARNING: Insert logic duplicated in save_interaction_atomic_async.
        # If you update SQL here, please update it there too.
        
        try:
            conversation_bk = await self._get_conversation_bk_async(user_key, channel_id, thread_ts)
            kh_conversation = self._generate_hash_key(conversation_bk)

            check_query = text(f"""
                SELECT 1 FROM {self.schema}.hub_conversation 
                WHERE kh_conversation = :kh_conversation
            """)

            result = await self._exec_async(
                check_query, {"kh_conversation": kh_conversation}, fetch=True
            )
            
            # Preparar datos del link (siempre necesarios)
            kh_user = self._generate_hash_key(user_key)
            kh_agent = self._generate_hash_key(self.agent_id)
            link_bk = f"{user_key}_{self.agent_id}_{conversation_bk}"
            kh_link = self._generate_hash_key(link_bk)
            
            if result:
                # Conversación existe, pero verificar si el link existe
                await self._ensure_link_exists_async(
                    kh_link, kh_user, kh_agent, kh_conversation
                )
                return kh_conversation

            insert_hub = text(f"""
                INSERT INTO {self.schema}.hub_conversation 
                (kh_conversation, ax_conversation, ct_ingest_dt, ax_src_system_datastore, ai_src_system)
                VALUES (:kh_conversation, :ax_conversation, :ct_ingest_dt, :ax_src_system, :ai_src_system)
            """)

            await self._exec_async(
                insert_hub,
                {
                    "kh_conversation": kh_conversation,
                    "ax_conversation": conversation_bk,
                    "ct_ingest_dt": datetime.now(),
                    "ax_src_system": self.src_system,
                    "ai_src_system": self.src_system_id,
                },
                commit=True,
            )

            insert_link = text(f"""
                INSERT INTO {self.schema}.lnk_users_agents_conversation 
                (kh_user_agent_conversation, kh_user, kh_agent, kh_conversation, 
                 ct_ingest_dt, ax_src_system_datastore, ai_src_system)
                VALUES (:kh_link, :kh_user, :kh_agent, :kh_conversation,
                        :ct_ingest_dt, :ax_src_system, :ai_src_system)
            """)

            await self._exec_async(
                insert_link,
                {
                    "kh_link": kh_link,
                    "kh_user": kh_user,
                    "kh_agent": kh_agent,
                    "kh_conversation": kh_conversation,
                    "ct_ingest_dt": datetime.now(),
                    "ax_src_system": self.src_system,
                    "ai_src_system": self.src_system_id,
                },
                commit=True,
            )

            logger.info(f"Conversation {conversation_bk} created")
            return kh_conversation

        except Exception as e:
            logger.error(f"Error getting/creating conversation: {e}")
            return None

    # ==================== MENSAJES (CHATS) ====================

    async def save_message_async(
        self,
        user_key: str,
        channel_id: str,
        message_type: str,
        content: str,
        thread_ts: str = None,
        feedback: bool = None,
        kh_conversation: bytes = None,
    ) -> bool:
        """Guarda un mensaje en sat_compass_current_chat y sat_compass_historical_chats (async)."""
        
        # WARNING: Insert logic duplicated in save_interaction_atomic_async.
        # If you update SQL here, please update it there too.
        
        try:
            if kh_conversation is None:
                kh_conversation = await self.get_or_create_conversation_async(
                    user_key, channel_id, thread_ts
                )

            conversation_bk = await self._get_conversation_bk_async(user_key, channel_id, thread_ts)
            link_bk = f"{user_key}_{self.agent_id}_{conversation_bk}"
            kh_link = self._generate_hash_key(link_bk)

            chat_data = {"content": content, "type": message_type}
            checksum = self._generate_checksum(chat_data)
            now = datetime.now()

            params = {
                "kh_link": kh_link,
                "ct_valid_from_dt": now,
                "ct_ingest_dt": now,
                "ax_src_system": self.src_system,
                "ah_checksum": checksum,
                "ax_message_type": message_type,
                "ab_feedback": feedback,
                "ax_content": content,
                "aj_attachments": json.dumps({}),
            }

            insert_current = text(f"""
                INSERT INTO {self.schema}.sat_compass_current_chat 
                (kh_user_agent_conversation, ct_valid_from_dt, ct_ingest_dt, 
                 ax_src_system_datastore, ah_checksum, ax_message_type, 
                 ab_feedback, ax_content, aj_attachments)
                VALUES (:kh_link, :ct_valid_from_dt, :ct_ingest_dt,
                        :ax_src_system, :ah_checksum, :ax_message_type,
                        :ab_feedback, :ax_content, :aj_attachments)
            """)
            await self._exec_async(insert_current, params, commit=True)

            insert_historical = text(f"""
                INSERT INTO {self.schema}.sat_compass_historical_chats 
                (kh_user_agent_conversation, ct_valid_from_dt, ct_ingest_dt, 
                 ax_src_system_datastore, ah_checksum, ax_message_type, 
                 ab_feedback, ax_content, aj_attachments)
                VALUES (:kh_link, :ct_valid_from_dt, :ct_ingest_dt,
                        :ax_src_system, :ah_checksum, :ax_message_type,
                        :ab_feedback, :ax_content, :aj_attachments)
            """)
            await self._exec_async(insert_historical, params, commit=True)

            logger.info(f"Message saved to compass tables: {message_type}")
            return True

        except Exception as e:
            logger.error(f"Error saving message: {e}")
            return False

    # ==================== CHUNKS (KNOWLEDGE) ====================

    def get_retrieved_chunks_metadata(self, retrieved_docs: List[Any]) -> List[Dict]:
        """
        Obtiene metadata de chunks recuperados.
        """
        try:
            chunks_info = []

            for doc in retrieved_docs:
                chunk_id = doc.metadata.get("ax_sub_sequence") or doc.metadata.get("id")

                chunks_info.append(
                    {
                        "chunk_id": chunk_id,
                        "content": doc.page_content[:200],
                        "score": getattr(doc, "score", None)
                        or doc.metadata.get("score"),
                        "metadata": doc.metadata,
                    }
                )

            logger.info(f"Retrieved {len(chunks_info)} chunks metadata")
            return chunks_info

        except Exception as e:
            logger.error(f"Error getting chunks metadata: {e}")
            return []

    # ==================== FEEDBACK ====================

    async def save_feedback_async(
        self,
        user_key: str,
        channel_id: str,
        thread_ts: str,
        feedback_positive: bool,
        reaction_emoji: str = None,
        comment: str = None,
    ) -> bool:
        """Actualiza el último mensaje con feedback en ambas tablas (async)."""
        try:
            conversation_bk = await self._get_conversation_bk_async(user_key, channel_id, thread_ts)
            link_bk = f"{user_key}_{self.agent_id}_{conversation_bk}"
            kh_link = self._generate_hash_key(link_bk)

            logger.info(f"save_feedback: user_key={user_key}, link_bk={link_bk}")

            feedback_data = {
                "feedback_positive": feedback_positive,
                "reaction_emoji": reaction_emoji,
                "timestamp": datetime.now().isoformat(),
            }
            if comment:
                feedback_data["comment"] = comment

            # Helper async para actualizar tabla
            async def update_table_async(table_name):
                query = text(f"""
                    UPDATE {self.schema}.{table_name}
                    SET ab_feedback = :ab_feedback,
                        aj_attachments = CAST(:aj_attachments AS jsonb)
                    WHERE kh_user_agent_conversation = :kh_link
                    AND ax_message_type = 'bot_response'
                    AND ct_valid_from_dt = (
                        SELECT MAX(ct_valid_from_dt) 
                        FROM {self.schema}.{table_name}
                        WHERE kh_user_agent_conversation = :kh_link 
                        AND ax_message_type = 'bot_response'
                    )
                """)
                await self._exec_async(
                    query,
                    {
                        "ab_feedback": feedback_positive,
                        "aj_attachments": json.dumps(feedback_data),
                        "kh_link": kh_link,
                    },
                    commit=True,
                )

            await update_table_async("sat_compass_current_chat")
            await update_table_async("sat_compass_historical_chats")

            logger.info(
                f"Feedback saved to compass tables: {'positive' if feedback_positive else 'negative'}"
            )
            return True

        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            return False

    # ==================== ANALYTICS ====================

    async def get_conversation_history_async(
        self, user_key: str, channel_id: str, thread_ts: str = None, limit: int = 50
    ) -> List[Dict]:
        """Obtiene historial de mensajes desde sat_compass_current_chat (async)."""
        try:
            conversation_bk = await self._get_conversation_bk_async(user_key, channel_id, thread_ts)
            link_bk = f"{user_key}_{self.agent_id}_{conversation_bk}"
            kh_link = self._generate_hash_key(link_bk)

            query = text(f"""
                SELECT 
                    ax_message_type,
                    ax_content,
                    ab_feedback,
                    ct_valid_from_dt
                FROM {self.schema}.sat_compass_current_chat
                WHERE kh_user_agent_conversation = :kh_link
                ORDER BY ct_valid_from_dt DESC
                LIMIT :limit
            """)

            rows = await self._exec_async(query, {"kh_link": kh_link, "limit": limit}, fetch=True) or []
            messages = [
                {
                    "message_type": row[0],
                    "content": row[1],
                    "feedback": row[2],
                    "timestamp": row[3],
                }
                for row in rows
            ]
            messages.reverse()
            return messages

        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
