-- =========================================================================================
-- ÍNDICES RECOMENDADOS PARA COMPASS CHATBOT (DATA VAULT 2.0)
-- Objetivo: Optimizar queries de historial, feedback y estrategia de sesión.
-- =========================================================================================

-- 1. Optimización de Historial de Conversación (get_conversation_history_async)
-- Query actual: SELECT ... WHERE kh_user_agent_conversation = :kh_link ORDER BY ct_valid_from_dt DESC LIMIT :limit
-- Impacto: Elimina sort en memoria y full scan en la tabla de chats.
CREATE INDEX IF NOT EXISTS idx_sccc_link_ts_desc
ON multiagent_rag_model.sat_compass_current_chat (kh_user_agent_conversation, ct_valid_from_dt DESC);

-- 2. Optimización de Feedback (save_feedback_async)
-- Query actual: UPDATE ... WHERE kh_user_agent_conversation = :kh_link AND ax_message_type = 'bot_response' ...
-- Impacto: Acelera la búsqueda del último mensaje del bot para actualizar feedback.
CREATE INDEX IF NOT EXISTS idx_sccc_link_type_ts
ON multiagent_rag_model.sat_compass_current_chat (kh_user_agent_conversation, ax_message_type, ct_valid_from_dt DESC);

CREATE INDEX IF NOT EXISTS idx_schc_link_type_ts
ON multiagent_rag_model.sat_compass_historical_chats (kh_user_agent_conversation, ax_message_type, ct_valid_from_dt DESC);

-- 3. Optimización de SessionBasedStrategy (get_last_conversation_async)
-- Query actual: JOINs complejos + LIKE 'pattern%' + ORDER BY last_activity DESC
-- Impacto: Permite index scan para buscar conversaciones recientes de un usuario/canal.

-- Índice en el HUB para búsquedas por patrón (LIKE 'user_channel_%')
-- Usamos operador text_pattern_ops para optimizar LIKE con prefijo.
CREATE INDEX IF NOT EXISTS idx_hub_conversation_ax_pattern
ON multiagent_rag_model.hub_conversation (ax_conversation text_pattern_ops);

-- Índices en LINK para acelerar los JOINs
CREATE INDEX IF NOT EXISTS idx_lnk_uac_kh_conversation
ON multiagent_rag_model.lnk_users_agents_conversation (kh_conversation);

CREATE INDEX IF NOT EXISTS idx_lnk_uac_kh_link
ON multiagent_rag_model.lnk_users_agents_conversation (kh_user_agent_conversation);

-- Índice en SAT para obtener la última actividad (ct_ingest_dt) rápidamente
CREATE INDEX IF NOT EXISTS idx_sccc_link_ingest_dt
ON multiagent_rag_model.sat_compass_current_chat (kh_user_agent_conversation, ct_ingest_dt DESC);

-- 4. Índices Únicos Parciales (Opcional pero recomendado para integridad)
-- Garantiza que solo haya una fila marcada como current (ai_current_flag = 1) por hash key.
-- Evita duplicados lógicos si falla la transacción o hay race conditions extremas.

CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_current_user
ON multiagent_rag_model.sat_compass_users_data (kh_user)
WHERE ai_current_flag = 1;

CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_current_agent
ON multiagent_rag_model.sat_agents_data (kh_agent)
WHERE ai_current_flag = 1;
