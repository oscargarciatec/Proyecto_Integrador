CREATE database spin-voyager;
CREATE user lg_process with password 'proyecto_integrador';
CREATE SCHEMA IF NOT EXISTS multiagent_rag_model;
ALTER SCHEMA multiagent_rag_model OWNER TO postgres;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE EXTENSION IF NOT EXISTS unaccent WITH SCHEMA public;
COMMENT ON EXTENSION unaccent IS 'text search dictionary that removes accents';

CREATE EXTENSION IF NOT EXISTS unaccent;
-- 2. Asegurar que la configuración "spanish_unaccent" exista y esté limpia
CREATE TEXT SEARCH CONFIGURATION public.spanish_unaccent ( COPY = pg_catalog.spanish );
ALTER TEXT SEARCH CONFIGURATION public.spanish_unaccent
    ALTER MAPPING FOR hword, hword_part, word
    WITH unaccent, spanish_stem;

DROP TABLE IF EXISTS multiagent_rag_model.hub_agents;
CREATE TABLE multiagent_rag_model.hub_agents
(
	kh_agent           BYTEA  NOT NULL PRIMARY KEY,
	ax_agent           VARCHAR  NOT NULL ,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ai_src_system        INTEGER  NULL
);

DROP TABLE IF EXISTS multiagent_rag_model.hub_conversation;
CREATE TABLE multiagent_rag_model.hub_conversation
(
	kh_conversation     BYTEA  NOT NULL PRIMARY KEY,
	ax_conversation     VARCHAR  NOT NULL ,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ai_src_system        INTEGER  NULL
);

DROP TABLE IF EXISTS multiagent_rag_model.hub_knowledge;
CREATE TABLE multiagent_rag_model.hub_knowledge
(
	kh_knowledge        BYTEA  NOT NULL PRIMARY KEY,
	ax_knowledge        VARCHAR  NOT NULL ,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ai_src_system        INTEGER  NULL
);


DROP TABLE IF EXISTS multiagent_rag_model.hub_users;
CREATE TABLE multiagent_rag_model.hub_users
(
	kh_user            BYTEA  NOT NULL PRIMARY KEY,
	ax_user            VARCHAR  NOT NULL ,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ai_src_system        INTEGER  NULL
);


DROP TABLE IF EXISTS multiagent_rag_model.lnk_agents_interactions;
CREATE TABLE multiagent_rag_model.lnk_agents_interactions
(
	kh_agents_interaction      BYTEA  NOT NULL PRIMARY KEY,
	kh_agent_main       BYTEA  NOT NULL,
	kh_agent_sec        BYTEA  NOT NULL,
	kh_conversation     BYTEA  NOT NULL,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ai_src_system        INTEGER  NULL
);

DROP TABLE IF EXISTS multiagent_rag_model.lnk_agents_knowledge;
CREATE TABLE multiagent_rag_model.lnk_agents_knowledge
(
	kh_agents_knowledge BYTEA  NOT NULL PRIMARY KEY,
	kh_agent            BYTEA  NOT NULL ,
	kh_knowledge        BYTEA  NOT NULL,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ai_src_system        INTEGER  NULL
);

DROP TABLE IF EXISTS multiagent_rag_model.lnk_context;
CREATE TABLE multiagent_rag_model.lnk_context
(
	kh_context          BYTEA  NOT NULL PRIMARY KEY,
	kh_user            BYTEA  NOT NULL ,
	kh_agent           BYTEA  NOT NULL,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ai_src_system        INTEGER  NULL
);

DROP TABLE IF EXISTS multiagent_rag_model.lnk_users_agents_conversation;
CREATE TABLE multiagent_rag_model.lnk_users_agents_conversation
(
	kh_user_agent_conversation     BYTEA  NOT NULL PRIMARY KEY,
	kh_user                        BYTEA  NOT NULL ,
	kh_agent                       BYTEA  NOT NULL ,
	kh_conversation                BYTEA  NOT NULL ,
	ct_ingest_dt                   TIMESTAMP  NOT NULL ,
	ax_src_system_datastore        VARCHAR  NULL ,
	ai_src_system                  INTEGER  NULL
);

DROP TABLE IF EXISTS multiagent_rag_model.lnk_retrieval_knowledge;
CREATE TABLE multiagent_rag_model.lnk_retrieval_knowledge
(
	kh_retrieval_knowledge BYTEA  NOT NULL PRIMARY KEY,
	kh_agent            BYTEA  NOT NULL ,
	kh_knowledge        BYTEA  NOT NULL,
	kh_user        BYTEA  NOT NULL,
	kh_conversation BYTEA  NOT NULL,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ai_src_system        INTEGER  NULL
);

DROP TABLE IF EXISTS multiagent_rag_model.sat_agents_data;
CREATE TABLE multiagent_rag_model.sat_agents_data
(
	kh_agent             BYTEA  NOT NULL ,
	ct_valid_from_dt     TIMESTAMP  NOT NULL ,
	ai_current_flag      INTEGER NOT  NULL ,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ah_checksum          BYTEA  NOT NULL ,
	ax_name              VARCHAR  NULL ,
	ax_description       VARCHAR  NULL ,
	ax_url               VARCHAR  NULL ,
	aj_agent_definition  JSON  NULL,
	aj_priming           JSON  NULL,
	aj_agent_examples    JSON  NULL,
	ab_is_supervisor     BOOLEAN NULL,
	PRIMARY KEY (kh_agent, ct_valid_from_dt)
) ;


DROP TABLE IF EXISTS multiagent_rag_model.sat_agents_interaction;
CREATE TABLE multiagent_rag_model.sat_agents_interaction
(
	kh_agents_interaction BYTEA  NOT NULL ,
	ct_valid_from_dt     TIMESTAMP  NOT NULL ,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ah_checksum          BYTEA  NOT NULL ,
	aj_input             JSON  NULL ,
	aj_ouput             JSON  NULL,
	PRIMARY KEY (kh_agents_interaction, ct_valid_from_dt)
) ;

DROP TABLE IF EXISTS multiagent_rag_model.sat_agents_knowledge;
CREATE TABLE multiagent_rag_model.sat_agents_knowledge
(
	kh_agents_knowledge BYTEA  NOT NULL ,
	ct_valid_from_dt     TIMESTAMP  NOT NULL ,
	ai_current_flag      INTEGER  NOT NULL ,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ah_checksum          BYTEA  NOT NULL ,
	cd_start_dt          DATE  NULL ,
	cd_end_dt            DATE  NULL ,
	PRIMARY KEY (kh_agents_knowledge, ct_valid_from_dt)
) ;


DROP TABLE IF EXISTS multiagent_rag_model.sat_compass_current_chat;
CREATE TABLE multiagent_rag_model.sat_compass_current_chat
(
	kh_user_agent_conversation BYTEA  NOT NULL ,
	ct_valid_from_dt     TIMESTAMP  NOT NULL ,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ah_checksum          BYTEA  NOT NULL ,
	ax_message_type      VARCHAR  NULL ,
	ab_feedback          boolean  NULL ,
	ax_content           VARCHAR  NULL ,
	aj_attachments       JSON  NULL ,
	PRIMARY KEY (kh_user_agent_conversation, ct_valid_from_dt)
) ;

DROP TABLE IF EXISTS multiagent_rag_model.sat_compass_historical_chats;
CREATE TABLE multiagent_rag_model.sat_compass_historical_chats
(
    kh_user_agent_conversation BYTEA  NOT NULL ,
	ct_valid_from_dt     TIMESTAMP  NOT NULL ,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ah_checksum          BYTEA  NOT NULL ,
	ax_message_type      VARCHAR  NULL ,
	ab_feedback          boolean  NULL,
	ax_content           VARCHAR  NULL ,
	aj_attachments       JSON  NULL ,
	PRIMARY KEY (kh_user_agent_conversation, ct_valid_from_dt)
) ;

DROP TABLE IF EXISTS multiagent_rag_model.sat_compass_context;
CREATE TABLE multiagent_rag_model.sat_compass_context
(
    kh_context          BYTEA NOT NULL ,
	ct_valid_from_dt     TIMESTAMP  NOT NULL ,
	ai_current_flag      INTEGER  NOT NULL ,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ah_checksum          BYTEA  NOT NULL ,
	ax_context           VARCHAR  NULL,
	PRIMARY KEY (kh_context, ct_valid_from_dt)
) 
;

DROP TABLE IF EXISTS multiagent_rag_model.sat_compass_users_data;
CREATE TABLE multiagent_rag_model.sat_compass_users_data
(
	kh_user            BYTEA  NOT NULL ,
	ct_valid_from_dt     TIMESTAMP  NOT NULL ,
	ai_current_flag      INTEGER  NOT NULL ,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ah_checksum          BYTEA  NOT NULL ,
	ax_display_nm        VARCHAR  NULL ,
	ax_user_context      VARCHAR  NULL ,
	ax_job_title              VARCHAR  NULL ,
	ax_email             VARCHAR  NULL,
	PRIMARY KEY (kh_user, ct_valid_from_dt)
) ;

DROP TABLE IF EXISTS multiagent_rag_model.sat_knowledge;
CREATE TABLE multiagent_rag_model.sat_knowledge
(
    kh_knowledge        BYTEA  NOT NULL ,
	ct_valid_from_dt     TIMESTAMP  NOT NULL ,
	ai_current_flag      INTEGER  NOT NULL ,
	ax_sub_sequence      VARCHAR  NOT NULL,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ah_checksum          BYTEA  NOT NULL ,
	ax_content           VARCHAR  NULL ,
	aa_embedding         VECTOR(768)  NULL,
	aa_fts_vector	   TSVECTOR  NULL ,
	PRIMARY KEY (kh_knowledge, ax_sub_sequence, ct_valid_from_dt)
) ;

ALTER TABLE multiagent_rag_model.sat_knowledge
ADD COLUMN ts_vector tsvector
GENERATED ALWAYS AS (to_tsvector('spanish_unaccent', ax_content)) STORED;
-- Crear indice
CREATE INDEX idx_sat_knowledge_ts_vector
ON multiagent_rag_model.sat_knowledge
USING gin (ts_vector);

DROP TABLE IF EXISTS multiagent_rag_model.sat_knowledge_metadata;
CREATE TABLE multiagent_rag_model.sat_knowledge_metadata
(
    kh_knowledge        BYTEA  NOT NULL ,
	ct_valid_from_dt     TIMESTAMP  NOT NULL ,
	ai_current_flag      INTEGER  NOT NULL ,
	ax_sub_sequence      VARCHAR  NOT NULL,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ah_checksum          BYTEA  NOT NULL ,
	ai_page_number        INTEGER  NULL ,
	ax_chunk_type		VARCHAR  NULL ,
	ax_content           VARCHAR  NULL ,
	ai_char_count        INTEGER  NULL ,
	ai_token_count       INTEGER  NULL ,
	ax_topic			VARCHAR  NULL ,
	ax_source			VARCHAR  NULL ,
	PRIMARY KEY (kh_knowledge, ax_sub_sequence, ct_valid_from_dt)
) ;


DROP TABLE IF EXISTS multiagent_rag_model.sat_retrieval_knowledge;
CREATE TABLE multiagent_rag_model.sat_retrieval_knowledge
(
    kh_retrieval_knowledge        BYTEA  NOT NULL ,
	ct_valid_from_dt     TIMESTAMP  NOT NULL ,
	ai_current_flag      INTEGER  NOT NULL ,
	ax_sub_sequence      VARCHAR  NOT NULL,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ah_checksum          BYTEA  NOT NULL ,
	ai_rank_chunk        INTEGER  NULL ,
	an_similarity_score		NUMERIC  NULL,
	PRIMARY KEY (kh_retrieval_knowledge, ax_sub_sequence, ct_valid_from_dt)
) ;

CREATE OR REPLACE FUNCTION multiagent_rag_model.update_fts_vector()
RETURNS TRIGGER AS $$
BEGIN
    -- This function now correctly uses the 'spanish' configuration
    -- for text processing, which understands Spanish grammar and stop words.
    NEW.aa_fts_vector := to_tsvector('spanish', NEW.ax_content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


--
-- Name: sat_knowledge_vector_update_trigger; Type: TRIGGER; Schema: multiagent_rag_model;
--
CREATE TRIGGER sat_knowledge_vector_update_trigger
BEFORE INSERT ON multiagent_rag_model.sat_knowledge
FOR EACH ROW EXECUTE FUNCTION multiagent_rag_model.update_fts_vector();

GRANT ALL ON SCHEMA multiagent_rag_model TO lg_process;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA multiagent_rag_model GRANT SELECT,INSERT,DELETE,UPDATE ON TABLES TO lg_process;
