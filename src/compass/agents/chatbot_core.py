# agents/chatbot_core.py
import asyncio
import os
import re
from datetime import date, datetime
from typing import Dict, Any, List, Optional

from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_google_alloydb_pg import (
    AlloyDBEngine,
    AlloyDBVectorStore,
    HybridSearchConfig,
    reciprocal_rank_fusion,
)
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from cachetools import TTLCache
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Imports para memoria conversacional (LangChain 1.0+)
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    # LangChain 1.0+ usa langchain_classic
    from langchain_classic.chains import (
        create_retrieval_chain,
    )
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain


from config.settings import ChatbotSettings
from database.data_vault_manager import DataVaultManager
from utils.pii_sanitizer import PIISanitizer
from utils.output_validator import OutputValidator
from prompts.rag_prompt import CONTEXTUALIZE_Q_PROMPT_TEMPLATE, QA_PROMPT_TEMPLATE
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ChatbotCore:
    """
    Handles the RAG logic, LLM chain execution, and logging.
    Usa inicialización async para VectorStore y Engine.
    """

    def __init__(self, settings: ChatbotSettings):
        """Constructor privado. Usa ChatbotCore.create() para instanciar."""
        self.settings = settings
        self.engine: Optional[AlloyDBEngine] = None
        self.persistence_engine = None  # Engine separado para persistencia
        self.vector_store: Optional[AlloyDBVectorStore] = None
        self.rag_chain: Optional[Runnable] = None
        self.persistence: Optional[DataVaultManager] = None
        self.ERROR_OUTPUT = "Lo sentimos, ha habido un problema! Ponte en contacto con los administradores :)"
        self._initialized = False
        # Cache para contexto personalizado de usuario (TTL 1 hora, max 500 usuarios)
        self.user_context_cache: TTLCache = TTLCache(maxsize=500, ttl=3600)

    @classmethod
    async def create(cls, settings: ChatbotSettings) -> "ChatbotCore":
        """Factory method async para crear una instancia inicializada."""
        instance = cls(settings)
        await instance._async_init()
        return instance

    @classmethod
    def create_sync(cls, settings: ChatbotSettings) -> "ChatbotCore":
        """Factory method síncrono (wrapper para entornos no-async como Flask)."""
        return asyncio.run(cls.create(settings))

    async def _async_init(self):
        """Inicialización async de todos los componentes."""
        logger.info(f"k_sim_search_num loaded: {self.settings.k_sim_search_num}")

        # Inicializar engine y vectorstore async
        self.engine = await self._initialize_engine_async()
        self.vector_store = await self._initialize_vector_store_async()
        self.rag_chain = self._initialize_rag_chain()

        # Engine separado para persistencia (evita conflictos con VectorStore)
        self.persistence_engine = await self._initialize_engine_async()

        # Inicializar persistencia con estrategia de sesión (ahora es async)
        timeout_hours = float(os.environ.get("SESSION_TIMEOUT_HOURS", "2"))

        agent_id = "Compass"
        agent_definition = {
            "llm_model_name": self.settings.llm_model_name,
            "embedding_model_name": self.settings.embedding_model_name,
            "vertex_ai_project_id": self.settings.vertex_ai_project_id,
            "vertex_ai_region": self.settings.vertex_ai_region,
        }
        self.persistence = DataVaultManager(
            engine=self.persistence_engine,
            agent_id=agent_id,
            agent_definition=agent_definition,
            agent_name="Compass-Slack",
            conversation_strategy="session",
            strategy_kwargs={"session_timeout_hours": timeout_hours},
        )
        # Inicializar agente de forma async
        await self.persistence.ensure_agent_exists_async()

        self._initialized = True
        logger.info("ChatbotCore initialized successfully (async).")

    async def _initialize_engine_async(self) -> AlloyDBEngine:
        """Inicializa el AlloyDB Engine de forma async.

        Soporta dos modos:
        1. Conexión directa por IP (cuando DB_HOST_IP está configurado) - más rápido
        2. Conexión por instancia (usa AlloyDB API para resolver IP)
        """
        try:
            if self.settings.db_host_ip:
                # Conexión directa usando SQLAlchemy async engine
                logger.info(f"Using direct IP connection: {self.settings.db_host_ip}")
                connection_string = (
                    f"postgresql+asyncpg://{self.settings.db_user}:{self.settings.db_password}"
                    f"@{self.settings.db_host_ip}:5432/{self.settings.db_name}"
                )
                async_engine = create_async_engine(
                    connection_string,
                    pool_size=20,
                    max_overflow=10,
                    pool_timeout=30,
                    pool_recycle=1800,
                    connect_args={
                        "timeout": 10,  # Timeout de conexión en segundos
                    },
                )
                # Pasar el loop actual para que _run_as_sync funcione
                loop = asyncio.get_running_loop()
                engine = AlloyDBEngine.from_engine(async_engine, loop=loop)

                # Verificar conexión con timeout
                logger.info("Testing database connection...")
                async with async_engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                logger.info("Database connection verified.")
            else:
                # Conexión por instancia (requiere AlloyDB API)
                logger.info("Using AlloyDB instance connection (API-based)")
                engine = await AlloyDBEngine.afrom_instance(
                    project_id=self.settings.gcp_project_id,
                    region=self.settings.db_region,
                    cluster=self.settings.db_cluster,
                    instance=self.settings.db_instance,
                    database=self.settings.db_name,
                    user=self.settings.db_user,
                    password=self.settings.db_password,
                    ip_type=self.settings.db_ip_type,
                )
            logger.info("AlloyDB Engine initialized (async).")
            return engine
        except Exception as e:
            logger.error(
                "CRITICAL: Failed to initialize Database Engine. Check configuration and credentials."
            )
            logger.error("Error type: %s", type(e).__name__)
            raise

    async def _initialize_vector_store_async(self) -> AlloyDBVectorStore:
        """Inicializa el AlloyDB Vector Store de forma async."""
        lc_embeddings = VertexAIEmbeddings(
            model_name=self.settings.embedding_model_name,
            project=self.settings.vertex_ai_project_id,
            location=self.settings.vertex_ai_region,
        )

        # CONFIGURACIÓN DE BÚSQUEDA HÍBRIDA
        # fetch_top_k: chunks a recuperar por cada método (vectorial y FTS) antes de fusión
        # rrf_k: constante para Reciprocal Rank Fusion (mayor valor = más peso a matches parciales)
        hybrid_search_config = HybridSearchConfig(
            tsv_column=self.settings.fts_vector_column_name,
            tsv_lang="public.spanish_unaccent",
            fusion_function=reciprocal_rank_fusion,
            fusion_function_parameters={
                "rrf_k": 60,
                "fetch_top_k": 80,
            },
        )

        # Usar método async create() en lugar de create_sync()
        vector_store = await AlloyDBVectorStore.create(
            engine=self.engine,
            schema_name=self.settings.db_schema_name,
            table_name=self.settings.collection_table_name,
            embedding_column=self.settings.embedding_column_name,
            id_column="ax_sub_sequence",
            content_column="ax_content",
            embedding_service=lc_embeddings,
            hybrid_search_config=hybrid_search_config,
            metadata_columns=["ai_current_flag"],
        )
        logger.info("AlloyDB VectorStore initialized (async).")
        return vector_store

    def _initialize_rag_chain(self) -> Runnable:
        """Sets up the RAG chain with conversational memory."""
        # LLM principal para respuestas de calidad
        llm = ChatVertexAI(
            model=self.settings.llm_model_name,
            project=self.settings.vertex_ai_project_id,
            location=self.settings.vertex_ai_region,
            temperature=0,
        )

        # LLM rápido para contextualización (solo extrae keywords)
        # gemini-2.0-flash-lite: económico y optimizado para tareas simples
        llm_fast = ChatVertexAI(
            model="gemini-2.0-flash-lite",
            project=self.settings.vertex_ai_project_id,
            location=self.settings.vertex_ai_region,
            temperature=0,
        )

        # Filtro como DICCIONARIO (Requerido por versiones recientes de la librería)
        filter_dict = {"ai_current_flag": 1}

        hybrid_search_config = HybridSearchConfig(
            tsv_column=self.settings.fts_vector_column_name,
            tsv_lang="public.spanish_unaccent",
            fusion_function=reciprocal_rank_fusion,
            fusion_function_parameters={
                "rrf_k": 60,
                "fetch_top_k": 80,
            },
        )

        section_pattern = re.compile(r"\b\d+(?:\.\d+){1,}\b")

        def _build_retriever(k: int):
            return self.vector_store.as_retriever(
                search_type="similarity",
                k=k,
                search_kwargs={
                    "k": k,
                    "filter": filter_dict,  # Se pasa el diccionario
                },
            )

        retriever = _build_retriever(self.settings.k_sim_search_num)

        # ---------------------------------------------------------
        # PROMPT DE CONTEXTUALIZACIÓN OPTIMIZADO PARA BÚSQUEDA
        # ---------------------------------------------------------
        # Este prompt reformula la pregunta del usuario para optimizar la búsqueda híbrida (vectorial + FTS).
        # Preserva elementos críticos como números de sección, referencias legales y valores numéricos.
        # ---------------------------------------------------------
        contextualize_q_prompt = CONTEXTUALIZE_Q_PROMPT_TEMPLATE

        # Usar LLM rápido para contextualización (tarea simple de keywords)
        contextualize_chain = contextualize_q_prompt | llm_fast | StrOutputParser()

        def _merge_docs(primary: List[Document], secondary: List[Document], k: int):
            merged: List[Document] = []
            seen: set[str] = set()

            def _doc_key(doc: Document) -> str:
                meta = getattr(doc, "metadata", {}) or {}
                return (
                    meta.get("ax_sub_sequence")
                    or meta.get("id")
                    or doc.page_content[:120]
                )

            for doc in primary + secondary:
                key = _doc_key(doc)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(doc)
                if len(merged) >= k:
                    break
            return merged

        async def _hybrid_similarity_search(query_text: str, k: int) -> List[Document]:
            if not query_text or len(query_text.strip()) < 3:
                return []
            return await self.vector_store.asimilarity_search(
                query_text,
                k=k,
                filter=filter_dict,
                hybrid_search_config=hybrid_search_config,
            )

        async def _fts_search(query_text: str, k: int) -> List[Document]:
            if not query_text or len(query_text.strip()) < 3:
                return []
            if not self.engine or not getattr(self.engine, "_pool", None):
                return []

            ts_query = query_text.strip()
            sql = text(
                f"""
                SELECT ax_sub_sequence, ax_content, ai_current_flag
                FROM {self.settings.db_schema_name}.{self.settings.collection_table_name}
                WHERE ai_current_flag = 1
                  AND {self.settings.fts_vector_column_name} @@ websearch_to_tsquery('public.spanish_unaccent', :ts_query)
                ORDER BY ts_rank(
                    {self.settings.fts_vector_column_name},
                    websearch_to_tsquery('public.spanish_unaccent', :ts_query)
                ) DESC
                LIMIT :k
                """
            )
            try:
                async with self.engine._pool.connect() as conn:
                    result = await conn.execute(sql, {"ts_query": ts_query, "k": k})
                    rows = result.fetchall()
            except Exception as exc:
                logger.warning("FTS retrieval failed; skipping FTS.", exc_info=exc)
                return []

            return [
                Document(
                    page_content=row[1],
                    metadata={
                        "ax_sub_sequence": row[0],
                        "ai_current_flag": row[2],
                        "retrieval": "fts",
                    },
                )
                for row in rows
            ]

        async def _retrieve_with_fallback(inputs: Dict[str, Any]):
            raw_query = await contextualize_chain.ainvoke(inputs)
            query = (raw_query or "").strip()
            user_query = (inputs.get("input") or "").strip()
            if not query:
                query = user_query
                logger.warning(
                    "Contextualized query was empty; falling back to original input."
                )
            else:
                # Log del query contextualizado para debugging
                logger.info(f"🔍 Contextualized query: '{query}'")
            if not query:
                return []
            effective_k = self.settings.k_sim_search_num
            if section_pattern.search(user_query):
                effective_k = max(effective_k, 12)
                logger.info(
                    f"Section-like query detected; using k={effective_k} for retrieval"
                )

            active_retriever = (
                retriever
                if effective_k == self.settings.k_sim_search_num
                else _build_retriever(effective_k)
            )
            composite_query = f"{query} {user_query}".strip()

            try:
                hybrid_docs = await _hybrid_similarity_search(
                    composite_query, effective_k
                )
            except Exception as exc:
                logger.warning(
                    "Hybrid search via AlloyDBVectorStore failed; falling back to manual hybrid.",
                    exc_info=exc,
                )
                hybrid_docs = []

            if hybrid_docs:
                logger.info(
                    "Hybrid search via AlloyDBVectorStore OK; docs=%s",
                    len(hybrid_docs),
                )
                return hybrid_docs

            similarity_docs = await active_retriever.ainvoke(composite_query)

            fts_query_text = composite_query
            if not fts_query_text:
                fts_query_text = user_query
            fts_docs = await _fts_search(fts_query_text, effective_k)
            if fts_docs:
                return _merge_docs(fts_docs, similarity_docs, effective_k)
            return similarity_docs

        history_aware_retriever = RunnableLambda(_retrieve_with_fallback)

        # Prompt del Sistema (QA) - El que responde al usuario final
        # NOTA: {user_context} se inyecta dinamicamente con las preferencias del usuario
        # Prompt del Sistema (QA) - El que responde al usuario final
        # NOTA: {user_context} se inyecta dinamicamente con las preferencias del usuario
        qa_prompt = QA_PROMPT_TEMPLATE

        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        return rag_chain

    def get_conversation_history(
        self, user_key: str, channel_id: str, thread_ts: str = None, limit: int = 10
    ) -> List:
        """Obtiene el historial de conversación desde la BD."""
        try:
            history_records = self.persistence.get_conversation_history(
                user_key=user_key,
                channel_id=channel_id,
                thread_ts=thread_ts,
                limit=limit,
            )

            chat_history = []
            for record in history_records:
                if record["message_type"] == "user_query":
                    chat_history.append(HumanMessage(content=record["content"]))
                elif record["message_type"] == "bot_response":
                    chat_history.append(AIMessage(content=record["content"]))

            return chat_history
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    async def get_user_context_async(self, user_email: str) -> Optional[str]:
        """
        Obtiene el contexto personalizado del usuario desde la BD.

        El contexto es generado periodicamente por compass_context_generator
        y contiene preferencias de comunicacion del usuario (tono, formato, etc.)

        Args:
            user_email: Email del usuario (ej: 'juan@spin.com')

        Returns:
            Texto con el contexto del usuario o None si no existe.
        """
        if not user_email or "@" not in user_email:
            return None

        # Verificar cache primero
        if user_email in self.user_context_cache:
            cached_value = self.user_context_cache[user_email]
            if cached_value is not None:
                logger.debug(f"User context cache HIT for {user_email}")
                return cached_value
            # Si es None cacheado, significa que ya verificamos y no hay contexto
            return None

        query = text("""
            SELECT cur.ax_context 
            FROM multiagent_rag_model.sat_compass_context cur
            INNER JOIN multiagent_rag_model.lnk_context lctx 
                ON cur.kh_context = lctx.kh_context
            INNER JOIN multiagent_rag_model.sat_compass_users_data usr 
                ON usr.kh_user = lctx.kh_user 
                AND usr.ax_src_system_datastore = lctx.ax_src_system_datastore
            WHERE usr.ax_email = :email 
              AND cur.ai_current_flag = 1 
              AND usr.ai_current_flag = 1
            LIMIT 1
        """)

        try:
            # Usar el metodo interno del engine para ejecutar queries async
            async with self.persistence_engine._pool.connect() as conn:
                result = await conn.execute(query, {"email": user_email})
                row = result.fetchone()
                if row and row[0]:
                    context = row[0]
                    # Guardar en cache
                    self.user_context_cache[user_email] = context
                    logger.info(f"User context loaded and cached for {user_email}")
                    return context
                else:
                    # Usuario sin contexto aun - guardar None en cache para evitar queries repetidas
                    self.user_context_cache[user_email] = None
                    logger.debug(f"No context found for {user_email} (cached as None)")
                    return None
        except Exception as e:
            logger.warning(f"Error fetching user context for {user_email}: {e}")
            return None

    async def get_chatbot_msg_async(
        self,
        user_query: str,
        user_key: str = None,
        channel_id: str = None,
        thread_ts: str = None,
    ) -> tuple[str, List[Document]]:
        """Invoca la cadena RAG de forma async con memoria conversacional."""
        try:
            # ===== OPERACIONES PARALELAS =====
            # La carga de historial, contexto de usuario y sanitizacion son independientes,
            # asi que las ejecutamos en paralelo para ahorrar tiempo.

            # 1. Iniciar carga de historial en background (si aplica)
            history_task = None
            if user_key and channel_id:
                history_task = asyncio.create_task(
                    self.persistence.get_conversation_history_async(
                        user_key=user_key,
                        channel_id=channel_id,
                        thread_ts=thread_ts,
                        limit=6,  # Limitar a 3 pares Q&A para evitar contexto muy largo
                    )
                )

            # 2. Iniciar carga de contexto personalizado del usuario (en paralelo)
            context_task = None
            if user_key:
                context_task = asyncio.create_task(
                    self.get_user_context_async(user_key)
                )

            # 3. Sanitizar PII mientras se cargan historial y contexto (operacion sync rapida)
            sanitization_result = PIISanitizer.sanitize(
                user_query, redact_amounts=True, new_schema=True
            )
            sanitized_query = sanitization_result.sanitized_text
            extracted_pii_values = list(sanitization_result.tokens.values())

            if sanitization_result.has_pii:
                logger.info(
                    f"PII detected and sanitized: {sanitization_result.pii_found}"
                )

            # 4. Esperar historial si se inicio la tarea
            chat_history = []
            if history_task:
                history_records = await history_task
                # Convertir dicts a objetos Message de LangChain
                for record in history_records:
                    content = record.get("content") or ""
                    # Saltar mensajes vacios o muy cortos
                    if not content or len(content.strip()) < 2:
                        continue
                    # Truncar contenido muy largo para evitar problemas con embeddings
                    content = content.strip()[:2000]
                    if record["message_type"] == "user_query":
                        chat_history.append(HumanMessage(content=content))
                    elif record["message_type"] == "bot_response":
                        chat_history.append(AIMessage(content=content))
                logger.info(f"Chat history ({len(chat_history)} messages) loaded.")
            else:
                logger.info("No user_id/channel_id provided, using empty history")

            # 5. Esperar contexto personalizado del usuario
            user_context_text = ""
            if context_task:
                user_context = await context_task
                if user_context:
                    user_context_text = f"""### Perfil de comunicacion de este Spinner:
{user_context}

Adapta tu tono, formato y nivel de detalle segun estas preferencias.
---"""
                    logger.info(f"User personalization context applied for {user_key}")

            # Log seguro (sin PII)
            logger.info(f"User Query (sanitized): {sanitized_query}")

            # Validar que el query no este vacio
            if not user_query or not user_query.strip():
                logger.warning("Empty user query received")
                return "Por favor, escribe tu pregunta.", []

            # Invocar la cadena RAG con query sanitizado y contexto de usuario
            result = await self.rag_chain.ainvoke(
                {
                    "input": sanitized_query.strip(),
                    "chat_history": chat_history,
                    "user_context": user_context_text,
                }
            )

            output = result.get("answer", "")
            retrieved_docs = result.get("context", [])

            # ===== CAPA 5: VALIDACIÓN DE OUTPUT =====
            # Verificar que el LLM no repita PII del usuario
            validation_result = OutputValidator.validate(
                output=output, user_input=user_query, extracted_pii=extracted_pii_values
            )

            if not validation_result.is_safe:
                logger.warning(
                    f"⚠️ PII detected in output, sanitizing: {validation_result.violations}"
                )
                output = validation_result.sanitized_output

            # Logging de documentos recuperados
            logger.info(f"📄 Retrieved {len(retrieved_docs)} chunks:")
            for i, doc in enumerate(retrieved_docs, 1):
                content_preview = doc.page_content[:100].replace("\n", " ")
                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                logger.info(f"  Chunk {i}: {content_preview}... | Meta: {metadata}")

            return output, retrieved_docs
        except Exception as e:
            logger.exception(f"FATAL ERROR: RAG chain failed to execute. Error: {e}")
            logger.error(
                f"Debug - Query: '{user_query}', History length: {len(chat_history)}"
            )
            # Si falla con historial, intentar sin historial
            if chat_history:
                logger.info("Retrying without chat history...")
                try:
                    result = await self.rag_chain.ainvoke(
                        {"input": user_query.strip(), "chat_history": []}
                    )
                    output = result.get("answer", "")
                    retrieved_docs = result.get("context", [])
                    return output, retrieved_docs
                except Exception as e2:
                    logger.exception(f"Retry also failed: {e2}")
            return self.ERROR_OUTPUT, []

    def get_chatbot_msg(
        self,
        user_query: str,
        user_key: str = None,
        channel_id: str = None,
        thread_ts: str = None,
    ) -> tuple[str, List[Document]]:
        """Wrapper síncrono para get_chatbot_msg_async (compatibilidad con Flask/Slack Bolt)."""
        return self._run_async(
            self.get_chatbot_msg_async(user_query, user_key, channel_id, thread_ts)
        )

    def _run_async(self, coro):
        """Ejecuta una coroutine usando el loop del engine."""
        if self.engine:
            return self.engine._run_as_sync(coro)
        # Fallback si engine no está disponible
        return asyncio.get_event_loop().run_until_complete(coro)

    def _extract_slack_user_data(
        self, slack_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fusiona la info del usuario proveniente de Slack (evento y metadatos)."""
        user_data: Dict[str, Any] = {}

        meta_user = slack_metadata.get("user")
        event_user = slack_metadata.get("event", {}).get("user")

        # Priorizar el payload del evento (es el que modificamos en slack_bot_async)
        for candidate in (event_user, meta_user):
            if isinstance(candidate, dict):
                user_data.update(candidate)

        if not user_data:
            # Último recurso: solo tenemos el ID como string
            user_id = None
            if isinstance(meta_user, str):
                user_id = meta_user
            elif isinstance(event_user, str):
                user_id = event_user
            if user_id:
                user_data["id"] = user_id

        # Mezclar perfiles adicionales que vienen fuera del objeto principal
        profile_sources: List[Dict[str, Any]] = []
        if isinstance(user_data.get("profile"), dict):
            profile_sources.append(user_data["profile"])
        event_profile = slack_metadata.get("event", {}).get("user_profile")
        if isinstance(event_profile, dict):
            profile_sources.append(event_profile)

        merged_profile: Dict[str, Any] = {}
        for profile in profile_sources:
            merged_profile.update(profile)

        if merged_profile:
            user_data["profile"] = merged_profile
            user_data.setdefault("real_name", merged_profile.get("real_name"))
            user_data.setdefault("email", merged_profile.get("email"))
            display_name = merged_profile.get("display_name") or merged_profile.get(
                "real_name"
            )
            if display_name:
                user_data.setdefault("display_name", display_name)
            user_data.setdefault("team_id", merged_profile.get("team"))
            username = merged_profile.get("display_name") or merged_profile.get(
                "real_name"
            )
            if username:
                user_data.setdefault("username", username)

        if not user_data.get("display_name"):
            fallback_name = (
                user_data.get("real_name")
                or user_data.get("username")
                or user_data.get("name")
            )
            if fallback_name:
                user_data["display_name"] = fallback_name

        return user_data

    async def save_interaction_async(
        self,
        slack_metadata: Dict[str, Any],
        query: str,
        output: str,
        retrieved_docs: List[Document],
    ) -> Dict[str, Any]:
        """Guarda una interacción completa en AlloyDB (async)."""
        try:
            # ===== SANITIZAR ANTES DE PERSISTIR =====
            # El query que se guarda en BD no debe contener PII
            sanitized_query = PIISanitizer.get_safe_log_text(query)
            # El output ya fue validado en get_chatbot_msg_async, pero doble check
            sanitized_output = OutputValidator.validate_and_fix(output, query)

            slack_user = self._extract_slack_user_data(slack_metadata)
            user_id = slack_user.get("id") or slack_user.get("user")

            if not user_id:
                logger.warning("No user_id found in slack_metadata")
                return {}

            if not slack_user.get("email"):
                logger.warning(
                    "Slack metadata is missing email for user %s; persistence will fallback to Slack ID",
                    user_id,
                )

            kh_user, user_key = await self.persistence.upsert_user_async(slack_user)

            if not user_key:
                logger.error("Failed to determine user_key")
                return {}

            channel_id = slack_metadata.get("channel") or slack_metadata.get(
                "event", {}
            ).get("channel")
            thread_ts = slack_metadata.get("thread_ts") or slack_metadata.get(
                "event", {}
            ).get("thread_ts")

            chunks_metadata = []
            if retrieved_docs:
                chunks_metadata = self.persistence.get_retrieved_chunks_metadata(
                    retrieved_docs
                )

            atomic_fn = getattr(self.persistence, "save_interaction_atomic_async", None)
            if callable(atomic_fn):
                result = await atomic_fn(
                    slack_user=slack_user,
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    query=sanitized_query,
                    output=sanitized_output,
                )

                if result:
                    logger.info(
                        f"Interaction saved (atomic): user={user_id}, chunks={len(chunks_metadata)}"
                    )
                    result["chunks_count"] = len(chunks_metadata)
                    return result

            kh_conversation = await self.persistence.get_or_create_conversation_async(
                user_key=user_key, channel_id=channel_id, thread_ts=thread_ts
            )

            query_saved = await self.persistence.save_message_async(
                user_key=user_key,
                channel_id=channel_id,
                message_type="user_query",
                content=sanitized_query,
                thread_ts=thread_ts,
                kh_conversation=kh_conversation,
            )

            response_saved = await self.persistence.save_message_async(
                user_key=user_key,
                channel_id=channel_id,
                message_type="bot_response",
                content=sanitized_output,
                thread_ts=thread_ts,
                feedback=None,
                kh_conversation=kh_conversation,
            )

            logger.info(
                f"Interaction saved: user={user_id}, chunks={len(chunks_metadata)}"
            )

            return {
                "kh_user": kh_user,
                "kh_conversation": kh_conversation,
                "query_saved": query_saved,
                "response_saved": response_saved,
                "chunks_count": len(chunks_metadata),
            }

        except Exception as e:
            logger.exception(f"Error saving interaction to database: {e}")
            return {}

    def chatbot_logging(
        self, message_metadata: Dict[str, Any], query: str, output: str
    ):
        """DEPRECADO: Legacy logging."""
        today = date.today()
        formatted_date = today.strftime("%Y-%m")
        log_path = self.settings.chatbot_logs_path
        print(f"[{datetime.now()}] Query: '{query}' | Response: '{output[:50]}...'")
        try:
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            with open(f"{log_path}{formatted_date}.log", "a") as fout:
                fout.write(f"------------------{datetime.now()}------------------\n")
                fout.write(f"Slack Metadata: {message_metadata}\n\n")
                fout.write(f"User Question: {query}\n\n")
                fout.write(f"Chatbot Output: {output}\n\n")
                fout.write("\n\n\n")
        except Exception as e:
            print(f"Error writing to local log file: {e}")
