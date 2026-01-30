# agents/slack_bot_async.py
"""
Versión async del bot de Slack usando AsyncApp.
Para uso con FastAPI/Starlette (ASGI).
"""

import re
import asyncio
import time
from collections import defaultdict
from slack_bolt.async_app import AsyncApp
from slack_sdk.errors import SlackApiError
from typing import Dict, Any

from cachetools import TTLCache

from config.settings import ChatbotSettings
from agents.chatbot_core import ChatbotCore
from utils.rate_limiter import RateLimiter
from utils.sensitive_detector import SensitiveContentDetector


class KeyedMutex:
    """
    Mutex que permite bloquear por clave (ej: conversation_id).
    Garantiza que los mensajes de una misma conversación se procesen en orden.
    """

    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def lock(self, key: str) -> asyncio.Lock:
        return self._locks[key]


class EventDeduplicator:
    """
    Deduplicador de eventos en memoria con TTL.
    Evita procesar el mismo event_id dos veces (Idempotencia).
    """

    def __init__(self, ttl_seconds: int = 600):
        self.processed_events: Dict[str, float] = {}
        self.ttl_seconds = ttl_seconds

    def is_processed(self, event_id: str) -> bool:
        self._cleanup()
        if event_id in self.processed_events:
            return True
        self.processed_events[event_id] = time.time()
        return False

    def _cleanup(self):
        now = time.time()
        keys_to_remove = [
            k for k, ts in self.processed_events.items() if now - ts > self.ttl_seconds
        ]
        for k in keys_to_remove:
            del self.processed_events[k]


def convert_markdown_to_slack_mrkdwn(text: str) -> str:
    """
    Convierte formato Markdown estándar a formato mrkdwn de Slack.

    Slack mrkdwn usa:
    - *texto* para negrita (no **texto**)
    - _texto_ para cursiva
    - ~texto~ para tachado
    - `codigo` para código inline
    - ```codigo``` para bloques de código
    - > para citas
    """
    if not text:
        return text

    # Proteger bloques de código (no modificar su contenido)
    code_blocks = []

    def protect_code_block(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    # Proteger bloques de código multilinea
    text = re.sub(r"```[\s\S]*?```", protect_code_block, text)
    # Proteger código inline
    text = re.sub(r"`[^`]+`", protect_code_block, text)

    # Convertir **texto** a *texto* (negrita)
    # Usar regex para manejar correctamente los asteriscos
    text = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", text)

    # Convertir __texto__ a _texto_ (cursiva alternativa en Markdown)
    text = re.sub(r"__([^_]+)__", r"_\1_", text)

    # Restaurar bloques de código
    for i, block in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{i}__", block)

    return text


class AsyncSlackChatbot(AsyncApp):
    """
    Slack Bot async usando slack_bolt.async_app.AsyncApp.
    Permite concurrencia real con FastAPI.
    """

    def __init__(self, settings: ChatbotSettings, core: ChatbotCore):
        super().__init__(
            token=settings.slack_bot_token,
            signing_secret=settings.slack_signing_secret,
        )
        self.core = core
        self.settings = settings
        self.ERROR_OUTPUT = self.core.ERROR_OUTPUT

        # Cache de usuarios con TTL y límite (punto 4: seguridad)
        self.user_cache: TTLCache = TTLCache(maxsize=1000, ttl=3600)  # 1 hora

        # Mecanismos de control de concurrencia
        self.keyed_mutex = KeyedMutex()
        self.deduplicator = EventDeduplicator(ttl_seconds=300)  # 5 minutos de memoria

        # Rate limiter para proteger contra abuso (punto 3: seguridad)
        self.rate_limiter = RateLimiter(
            max_requests=30,  # 30 mensajes por minuto
            window_seconds=60,
            burst_limit=5,  # Máximo 5 mensajes en 10 segundos
            burst_window_seconds=10,
        )

        self._register_handlers()

    def _register_handlers(self):
        """Registra todos los event listeners async."""
        # Mensaje directo al bot
        self.message(".*")(self._message_handler)

        # Mención del bot en canal
        self.event("app_mention")(self._handle_app_mention)

        # Reacciones para feedback
        self.event("reaction_added")(self._handle_feedback_reaction)

        # Botones de feedback
        self.action("feedback_positive")(self._handle_button_positive)
        self.action("feedback_negative")(self._handle_button_negative)
        self.view("feedback_comment_modal")(self._handle_feedback_comment)

        # Ignorar subtipos de mensaje (ediciones, etc.)
        @self.event("message")
        async def handle_message_events(event, logger):
            if event.get("subtype") is None:
                pass

    async def _get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Obtiene info del usuario con cache."""
        if user_id in self.user_cache:
            return self.user_cache[user_id]

        try:
            resp = await self.client.users_info(user=user_id)
            if resp["ok"]:
                user = resp["user"]
                profile = user.get("profile", {})
                user_info = {
                    "id": user["id"],
                    "name": user.get("name"),
                    "team_id": user.get("team_id"),
                    "email": profile.get("email"),
                    "real_name": profile.get("real_name"),
                    "username": user.get("name"),
                    "title": profile.get("title"),
                    "profile": profile,
                }
                self.user_cache[user_id] = user_info
                return user_info
        except Exception as e:
            print(f"Error fetching user info for {user_id}: {e}")

        return {"id": user_id}

    def _extract_query_from_mention(self, body: Dict[str, Any]) -> str | None:
        """Extrae la query del usuario de un app_mention."""
        try:
            if "blocks" in body["event"]:
                for block in body["event"]["blocks"]:
                    if block["type"] == "rich_text":
                        for element in block["elements"]:
                            if element["type"] == "rich_text_section":
                                text_elements = [
                                    e["text"]
                                    for e in element["elements"]
                                    if e["type"] == "text"
                                ]
                                if text_elements:
                                    return " ".join(text_elements).strip()
            text = body["event"].get("text", "")
            return re.sub(r"<@U[A-Z0-9]+>", "", text).strip()
        except Exception as e:
            print(f"Error extracting query from mention: {e}")
            return None

    async def _process_message(
        self, message: Dict[str, Any], client, say, is_mention: bool
    ):
        """Lógica compartida para mensajes y menciones (async)."""
        # 1. Agregar reacción de procesando
        channel = message["channel"] if not is_mention else message["event"]["channel"]
        ts = message["ts"] if not is_mention else message["event"]["ts"]

        await client.reactions_add(channel=channel, timestamp=ts, name="eyes")

        # 2. Extraer query e info de usuario
        if not is_mention:
            user_query = message.get("text")
            user_id = message.get("user")
            channel_id = message.get("channel")
            thread_ts = message.get("thread_ts")
        else:
            user_query = self._extract_query_from_mention(message)
            user_id = message.get("event", {}).get("user")
            channel_id = message.get("event", {}).get("channel")
            thread_ts = message.get("event", {}).get("thread_ts")

        if not user_query:
            await say(self.ERROR_OUTPUT)
            return

        # Obtener info del usuario (async)
        user_info = await self._get_user_info(user_id)
        user_key = user_info.get("email") or user_id

        # 3. Obtener respuesta del RAG (ASYNC!)
        output, retrieved_docs = await self.core.get_chatbot_msg_async(
            user_query=user_query,
            user_key=user_key,
            channel_id=channel_id,
            thread_ts=thread_ts,
        )

        # 4. Guardar interacción (async - bloqueante para evitar problemas de CPU en Cloud Run)
        metadata_copy = message.copy()
        if is_mention:
            if "event" in metadata_copy:
                metadata_copy["event"] = metadata_copy["event"].copy()
                metadata_copy["event"]["user"] = user_info
        else:
            metadata_copy["user"] = user_info

        await self.core.save_interaction_async(
            slack_metadata=metadata_copy,
            query=user_query,
            output=output,
            retrieved_docs=retrieved_docs,
        )

        # 5. Enviar respuesta con botones de feedback
        MAX_SLACK_CHARS = 2900

        # Convertir Markdown estándar a mrkdwn de Slack
        slack_output = convert_markdown_to_slack_mrkdwn(output)

        if len(slack_output) <= MAX_SLACK_CHARS:
            response_args = {
                "text": slack_output,
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": slack_output},
                    },
                    {"type": "divider"},
                    {
                        "type": "context",
                        "elements": [
                            {"type": "mrkdwn", "text": "_¿Te fue útil esta respuesta?_"}
                        ],
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "👍 Sí",
                                    "emoji": True,
                                },
                                "value": "positive",
                                "action_id": "feedback_positive",
                                "style": "primary",
                            },
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "👎 No",
                                    "emoji": True,
                                },
                                "value": "negative",
                                "action_id": "feedback_negative",
                                "style": "danger",
                            },
                        ],
                    },
                ],
            }
            if thread_ts:
                response_args["thread_ts"] = thread_ts
            bot_response = await say(**response_args)
            await self._remove_old_feedback_buttons(
                channel_id=channel_id,
                thread_ts=thread_ts,
                current_ts=bot_response.get("ts") if bot_response else None,
            )
        else:
            # Respuesta larga: dividir en chunks
            chunks = self._split_message(slack_output, MAX_SLACK_CHARS)

            first_args = {
                "text": chunks[0],
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": chunks[0]}}
                ],
            }
            if thread_ts:
                first_args["thread_ts"] = thread_ts
            bot_response = await say(**first_args)

            for chunk in chunks[1:]:
                await say(
                    text=chunk,
                    blocks=[
                        {"type": "section", "text": {"type": "mrkdwn", "text": chunk}}
                    ],
                    thread_ts=bot_response["ts"],
                )

            # Feedback buttons al final
            feedback_msg = await say(
                text="¿Te fue útil?",
                blocks=[
                    {
                        "type": "context",
                        "elements": [
                            {"type": "mrkdwn", "text": "_¿Te fue útil esta respuesta?_"}
                        ],
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "👍 Sí",
                                    "emoji": True,
                                },
                                "value": "positive",
                                "action_id": "feedback_positive",
                                "style": "primary",
                            },
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "👎 No",
                                    "emoji": True,
                                },
                                "value": "negative",
                                "action_id": "feedback_negative",
                                "style": "danger",
                            },
                        ],
                    },
                ],
                thread_ts=bot_response["ts"],
            )
            await self._remove_old_feedback_buttons(
                channel_id=channel_id,
                thread_ts=thread_ts,
                current_ts=feedback_msg.get("ts") if feedback_msg else None,
            )

        # 6. Quitar reacción
        try:
            await client.reactions_remove(channel=channel, timestamp=ts, name="eyes")
        except SlackApiError as exc:
            if exc.response.get("error") != "no_reaction":
                raise

    def _split_message(self, text: str, max_chars: int) -> list[str]:
        """Divide un mensaje largo en chunks."""
        chunks = []
        current = ""
        for line in text.split("\n"):
            if len(current) + len(line) + 1 <= max_chars:
                current += line + "\n"
            else:
                if current:
                    chunks.append(current.strip())
                current = line + "\n"
        if current:
            chunks.append(current.strip())
        return chunks

    async def _remove_old_feedback_buttons(
        self, channel_id: str, thread_ts: str | None, current_ts: str | None
    ):
        """
        Deja botones solo en el último mensaje del bot dentro del hilo/canal.
        Remueve el bloque de acciones y el texto de feedback en bot_responses anteriores.
        """
        if not current_ts:
            return
        try:

            def is_feedback_context(block: dict) -> bool:
                if block.get("type") != "context":
                    return False
                for el in block.get("elements", []):
                    if el.get(
                        "type"
                    ) == "mrkdwn" and "¿Te fue útil esta respuesta?" in (
                        el.get("text") or ""
                    ):
                        return True
                return False

            # Para threads, limitar la búsqueda a las replies; si no, al canal.
            if thread_ts:
                history = await self.client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts,
                    limit=10,
                    inclusive=False,
                    latest=current_ts,
                )
                messages = history.get("messages", [])
            else:
                history = await self.client.conversations_history(
                    channel=channel_id, limit=10, inclusive=False, latest=current_ts
                )
                messages = history.get("messages", [])

            for msg in messages:
                # Buscar el bot_response más reciente antes del actual
                if msg.get("ts") == current_ts:
                    continue
                if not (msg.get("bot_id") or msg.get("user") == "BOT"):
                    continue
                blocks = msg.get("blocks") or []
                has_actions = any(b.get("type") == "actions" for b in blocks)
                has_feedback_context = any(is_feedback_context(b) for b in blocks)
                if has_actions or has_feedback_context:
                    updated_blocks = [
                        b
                        for b in blocks
                        if b.get("type") != "actions" and not is_feedback_context(b)
                    ]
                    await self.client.chat_update(
                        channel=channel_id,
                        ts=msg["ts"],
                        text=msg.get("text") or "",
                        blocks=updated_blocks,
                    )
                    break
        except Exception as exc:
            print(f"Error removing old feedback buttons: {exc}")

    async def _run_background_processing(self, coro):
        """Helper para ejecutar tareas en background con logging de errores."""
        try:
            await coro
        except Exception as e:
            print(f"Error in background task: {e}")

    async def _process_sensitive_mention(
        self, body: Dict[str, Any], client, user_id: str
    ):
        """
        Procesa una mención con contenido sensible.
        Responde por DM en lugar de en el canal público.
        """
        try:
            print(f"[SENSITIVE] Starting DM process for user: {user_id}")

            # Abrir conversación DM con el usuario
            dm_response = await client.conversations_open(users=[user_id])
            dm_channel = dm_response["channel"]["id"]
            print(f"[SENSITIVE] DM channel opened: {dm_channel}")

            # Extraer query
            user_query = self._extract_query_from_mention(body)
            print(
                f"[SENSITIVE] Extracted query: {user_query[:50] if user_query else 'None'}..."
            )

            if not user_query:
                await client.chat_postMessage(
                    channel=dm_channel,
                    text=self.ERROR_OUTPUT,
                )
                print("[SENSITIVE] No query found, sent error message")
                return

            # Obtener info del usuario
            user_info = await self._get_user_info(user_id)
            user_key = user_info.get("email") or user_id
            print(f"[SENSITIVE] User key: {user_key}")

            # Obtener respuesta del RAG
            print("[SENSITIVE] Calling RAG...")
            output, retrieved_docs = await self.core.get_chatbot_msg_async(
                user_query=user_query,
                user_key=user_key,
                channel_id=dm_channel,
                thread_ts=None,
            )
            print(
                f"[SENSITIVE] RAG response received: {len(output)} chars, {len(retrieved_docs)} docs"
            )

            # Guardar interacción (async - bloqueante)
            event = body.get("event", {})
            metadata_copy = body.copy()
            if "event" in metadata_copy:
                metadata_copy["event"] = event.copy()
                metadata_copy["event"]["user"] = user_info

            await self.core.save_interaction_async(
                slack_metadata=metadata_copy,
                query=user_query,
                output=output,
                retrieved_docs=retrieved_docs,
            )
            print("[SENSITIVE] Interaction saved")

            # Enviar respuesta por DM (convertir Markdown a mrkdwn de Slack)
            slack_output = convert_markdown_to_slack_mrkdwn(output)
            dm_intro = (
                "🔒 *Respuesta privada a tu consulta en el canal:*\n\n"
                f"> _{user_query[:100]}{'...' if len(user_query) > 100 else ''}_\n\n"
            )
            full_message = dm_intro + slack_output

            # Incluir botones de feedback (consistente con _process_message)
            await client.chat_postMessage(
                channel=dm_channel,
                text=full_message,
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": full_message},
                    },
                    {"type": "divider"},
                    {
                        "type": "context",
                        "elements": [
                            {"type": "mrkdwn", "text": "_¿Te fue útil esta respuesta?_"}
                        ],
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "👍 Sí",
                                    "emoji": True,
                                },
                                "value": "positive",
                                "action_id": "feedback_positive",
                                "style": "primary",
                            },
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "👎 No",
                                    "emoji": True,
                                },
                                "value": "negative",
                                "action_id": "feedback_negative",
                                "style": "danger",
                            },
                        ],
                    },
                ],
            )
            print(f"[SENSITIVE] DM sent successfully to channel: {dm_channel}")

        except Exception as e:
            print(f"[SENSITIVE] ERROR processing sensitive mention: {e}")
            import traceback

            traceback.print_exc()

    async def _message_handler(
        self, message: Dict[str, Any], body: Dict[str, Any], client, say
    ):
        """Handler para mensajes directos."""
        event_id = body.get("event_id")
        if event_id and self.deduplicator.is_processed(event_id):
            return

        user_id = message.get("user")
        channel_id = message.get("channel")
        thread_ts = message.get("thread_ts")

        # Rate limiting check
        if user_id:
            allowed, error_msg = self.rate_limiter.is_allowed(user_id)
            if not allowed:
                await say(text=f"⏳ {error_msg}", thread_ts=thread_ts)
                return

        lock_key = f"{channel_id}:{thread_ts}" if thread_ts else channel_id

        async def process_wrapper():
            async with self.keyed_mutex.lock(lock_key):
                await self._process_message(message, client, say, is_mention=False)

        # Fire-and-forget: No esperamos a que termine el RAG
        asyncio.create_task(self._run_background_processing(process_wrapper()))

    async def _handle_app_mention(self, body: Dict[str, Any], client, say):
        """Handler para menciones del bot."""
        event_id = body.get("event_id")
        if event_id and self.deduplicator.is_processed(event_id):
            return

        event = body.get("event", {})
        user_id = event.get("user")
        channel_id = event.get("channel")
        thread_ts = event.get("thread_ts")
        message_text = event.get("text", "")

        # Rate limiting check
        if user_id:
            allowed, error_msg = self.rate_limiter.is_allowed(user_id)
            if not allowed:
                await say(text=f"⏳ {error_msg}", thread_ts=thread_ts)
                return

        # Detectar contenido sensible en canales públicos
        is_sensitive, _ = SensitiveContentDetector.is_sensitive(message_text)
        if is_sensitive and user_id:
            # Responder ephemerally y sugerir DM
            await client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text=SensitiveContentDetector.get_ephemeral_warning(),
            )
            # Procesar normalmente pero la respuesta irá por DM
            asyncio.create_task(
                self._run_background_processing(
                    self._process_sensitive_mention(body, client, user_id)
                )
            )
            return

        lock_key = f"{channel_id}:{thread_ts}" if thread_ts else channel_id

        async def process_wrapper():
            async with self.keyed_mutex.lock(lock_key):
                await self._process_message(body, client, say, is_mention=True)

        # Fire-and-forget: No esperamos a que termine el RAG
        asyncio.create_task(self._run_background_processing(process_wrapper()))

    async def _handle_feedback_reaction(self, body: Dict[str, Any], client):
        """Captura reacciones para feedback."""
        try:
            event = body.get("event", {})
            reaction = event.get("reaction")
            user_id = event.get("user")
            item = event.get("item", {})
            channel = item.get("channel")
            message_ts = item.get("ts")

            positive_reactions = [
                "thumbsup",
                "+1",
                "white_check_mark",
                "heavy_check_mark",
                "100",
                "clap",
                "raised_hands",
                "star",
                "heart",
            ]
            negative_reactions = [
                "thumbsdown",
                "-1",
                "x",
                "disappointed",
                "confused",
                "thinking_face",
                "question",
            ]

            if reaction in positive_reactions:
                feedback_positive = True
            elif reaction in negative_reactions:
                feedback_positive = False
            else:
                return

            message_info = await client.conversations_history(
                channel=channel, latest=message_ts, limit=1, inclusive=True
            )
            if not message_info.get("messages"):
                return

            msg = message_info["messages"][0]
            if msg.get("bot_id") or msg.get("user") == "BOT":
                user_info = await self._get_user_info(user_id)
                user_key = user_info.get("email") or user_id
                thread_ts = item.get("thread_ts")

                await self.core.persistence.save_feedback_async(
                    user_key=user_key,
                    channel_id=channel,
                    thread_ts=thread_ts,
                    feedback_positive=feedback_positive,
                    reaction_emoji=reaction,
                )
        except Exception as e:
            print(f"Error handling feedback reaction: {e}")

    async def _handle_button_positive(self, ack, body, client):
        """Handler para botón de feedback positivo."""
        await ack()
        try:
            user_id = body["user"]["id"]
            channel_id = body["channel"]["id"]
            message = body["message"]
            thread_ts = message.get("thread_ts")

            user_info = await self._get_user_info(user_id)
            user_key = user_info.get("email") or user_id

            await self.core.persistence.save_feedback_async(
                user_key=user_key,
                channel_id=channel_id,
                thread_ts=thread_ts,
                feedback_positive=True,
                reaction_emoji="button_thumbsup",
            )

            # Actualizar UI
            original_blocks = message.get("blocks", [])
            updated_blocks = [b for b in original_blocks if b.get("type") != "actions"]
            updated_blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "✅ _¡Gracias por tu feedback positivo!_",
                        }
                    ],
                }
            )
            await client.chat_update(
                channel=channel_id,
                ts=message["ts"],
                text=message["text"],
                blocks=updated_blocks,
            )
        except Exception as e:
            print(f"Error handling positive feedback: {e}")

    async def _handle_button_negative(self, ack, body, client):
        """Handler para botón de feedback negativo."""
        await ack()
        try:
            user_id = body["user"]["id"]
            channel_id = body["channel"]["id"]
            message = body["message"]
            thread_ts = message.get("thread_ts")

            # Abrir modal primero (trigger_id expira rápido)
            await client.views_open(
                trigger_id=body["trigger_id"],
                view={
                    "type": "modal",
                    "callback_id": "feedback_comment_modal",
                    "title": {"type": "plain_text", "text": "Cuéntanos más"},
                    "submit": {"type": "plain_text", "text": "Enviar"},
                    "close": {"type": "plain_text", "text": "Cancelar"},
                    "private_metadata": f"{user_id}|{channel_id}|{thread_ts or ''}",
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "Nos ayudaría saber qué podemos mejorar:",
                            },
                        },
                        {
                            "type": "input",
                            "block_id": "comment_block",
                            "element": {
                                "type": "plain_text_input",
                                "action_id": "comment_input",
                                "multiline": True,
                                "placeholder": {
                                    "type": "plain_text",
                                    "text": "¿Qué esperabas encontrar?",
                                },
                            },
                            "label": {
                                "type": "plain_text",
                                "text": "Comentario (opcional)",
                            },
                            "optional": True,
                        },
                    ],
                },
            )

            user_info = await self._get_user_info(user_id)
            user_key = user_info.get("email") or user_id

            await self.core.persistence.save_feedback_async(
                user_key=user_key,
                channel_id=channel_id,
                thread_ts=thread_ts,
                feedback_positive=False,
                reaction_emoji="button_thumbsdown",
            )

            # Actualizar UI
            original_blocks = message.get("blocks", [])
            updated_blocks = [b for b in original_blocks if b.get("type") != "actions"]
            updated_blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": "📝 _Gracias por tu feedback._"}
                    ],
                }
            )
            await client.chat_update(
                channel=channel_id,
                ts=message["ts"],
                text=message["text"],
                blocks=updated_blocks,
            )
        except Exception as e:
            print(f"Error handling negative feedback: {e}")

    async def _handle_feedback_comment(self, ack, body, client, view):
        """Handler para modal de comentarios."""
        await ack()
        try:
            metadata = view["private_metadata"].split("|")
            user_id = metadata[0]
            channel_id = metadata[1]
            thread_ts = metadata[2] if len(metadata) > 2 and metadata[2] else None

            comment = view["state"]["values"]["comment_block"]["comment_input"]["value"]

            if comment:
                user_info = await self._get_user_info(user_id)
                user_key = user_info.get("email") or user_id

                await self.core.persistence.save_feedback_async(
                    user_key=user_key,
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    feedback_positive=False,
                    reaction_emoji="button_thumbsdown_with_comment",
                    comment=comment,
                )

                await client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text="¡Gracias por tu comentario! Lo revisaremos.",
                )
        except Exception as e:
            print(f"Error handling feedback comment: {e}")
