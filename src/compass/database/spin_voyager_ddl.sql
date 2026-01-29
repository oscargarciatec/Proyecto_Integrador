CREATE database spin-voyager;
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

DROP TABLE IF EXISTS multiagent_rag_model.sat_current_chat;
CREATE TABLE multiagent_rag_model.sat_current_chat
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

DROP TABLE IF EXISTS multiagent_rag_model.sat_historical_chats;
CREATE TABLE multiagent_rag_model.sat_historical_chats
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

DROP TABLE IF EXISTS multiagent_rag_model.sat_context;
CREATE TABLE multiagent_rag_model.sat_context
(
    kh_context          BYTEA NOT NULL ,
	ct_valid_from_dt     TIMESTAMP  NOT NULL ,
	ai_current_flag      INTEGER  NOT NULL ,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ah_checksum          BYTEA  NOT NULL ,
	ax_context           VARCHAR  NULL,
	PRIMARY KEY (kh_context, ct_valid_from_dt)
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

DROP TABLE IF EXISTS multiagent_rag_model.sat_employees_data;
CREATE TABLE multiagent_rag_model.sat_employees_data
(
    kh_knowledge        BYTEA  NOT NULL ,
	ct_valid_from_dt     TIMESTAMP  NOT NULL ,
	ai_current_flag      INTEGER  NOT NULL ,
	ax_sub_sequence      VARCHAR  NOT NULL,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ah_checksum          BYTEA  NOT NULL ,
	ai_trip_request_id INTEGER,
	ai_employee_id	INTEGER,
	ax_employee_position	VARCHAR,
	ax_division_id	VARCHAR,
	ax_division_nm	VARCHAR,
	ax_direction	VARCHAR,
	ax_legal_entity_id	VARCHAR,
	ax_legal_entity_nm	VARCHAR,
	ax_hr_cost_center	VARCHAR,
	PRIMARY KEY (kh_knowledge, ct_valid_from_dt)
) 
;

DROP TABLE IF EXISTS multiagent_rag_model.sat_trips_general_data;
CREATE TABLE multiagent_rag_model.sat_trips_general_data
(
    kh_knowledge        BYTEA  NOT NULL ,
	ct_valid_from_dt     TIMESTAMP  NOT NULL ,
	ai_current_flag      INTEGER  NOT NULL ,
	ax_sub_sequence      VARCHAR  NOT NULL,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ah_checksum          BYTEA  NOT NULL ,
	ai_trip_request_id INTEGER,
	ai_trip_cd	INTEGER,
	ax_trip_subject	VARCHAR,
	ax_trip_justification	VARCHAR,
	ax_origin_city_nm	VARCHAR,
	ax_destination_city_nm	VARCHAR,
	ax_trip_type	VARCHAR,
	ax_air_transport_type	VARCHAR,
	ax_land_transport_type	VARCHAR,
	ab_trip_by_air	BOOLEAN,
	ab_trip_by_land	BOOLEAN,
	ai_trip_days	INTEGER,
	ax_trip_status	VARCHAR,
	cd_start_dt	DATE,
	cd_end_dt	DATE,
	cd_registry_dt	DATE,
	cd_submission_dt	DATE,
	cd_receipt_dt	DATE,
	cd_doc_digitalization_dt	DATE,
	ct_first_policy_validation_dt	TIMESTAMP,
	ct_second_policy_validation_dt	TIMESTAMP,
	ct_final_validation_dt	TIMESTAMP,
	cd_application_closing_dt	DATE,
	ct_doc_incidence_dt	TIMESTAMP,
	ct_policy_incidence_dt	TIMESTAMP,
	ax_approver_nm VARCHAR,
	PRIMARY KEY (kh_knowledge, ct_valid_from_dt)
) ;

DROP TABLE IF EXISTS multiagent_rag_model.sat_trips_expenses;
CREATE TABLE multiagent_rag_model.sat_trips_expenses
(
    kh_knowledge        BYTEA  NOT NULL ,
	ct_valid_from_dt     TIMESTAMP  NOT NULL ,
	ai_current_flag      INTEGER  NOT NULL ,
	ax_sub_sequence      VARCHAR  NOT NULL,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ah_checksum          BYTEA  NOT NULL ,
	ai_trip_request_id INTEGER,
	an_mileage_amt	NUMERIC,
	an_advance_amt	NUMERIC,
	an_total_deductible_expense_amt	NUMERIC,
	an_total_non_deductible_expense_amt	NUMERIC,
	an_total_non_deductible_expense_usd_amt	NUMERIC,
	an_total_expense_amt	NUMERIC,
	an_total_accommodation_amt	NUMERIC,
	an_total_fuel_amt	NUMERIC,
	an_food_expense_amt	NUMERIC,
	an_tolls_amt	NUMERIC,
	an_flight_tickets_amt	NUMERIC,
	an_flight_ticket_by_company_amt	NUMERIC,
	an_flight_ticket_by_passenger_amt	NUMERIC,
	an_flight_ticket_by_passenger_cash_card_amt	NUMERIC,
	an_flight_seat_amt	NUMERIC,
	an_flight_tickets_agency_commission_by_company_amt	NUMERIC,
	an_flight_tickets_agency_commission_by_passenger_amt	NUMERIC,
	an_commission_by_company_amt	NUMERIC,
	an_tua_amt	NUMERIC,
	an_tua_by_company_amt	NUMERIC,
	an_tua_by_passenger_amt	NUMERIC,
	an_flight_change_amt	NUMERIC,
	an_service_amt	NUMERIC,
	an_not_used_flight_tickets_by_passenger_amt	NUMERIC,
	an_business_lunch_expense_amt	NUMERIC,
	an_parking_amt	NUMERIC,
	an_fuel_amt	NUMERIC,
	an_accommodation_amt	NUMERIC,
	an_accommodation_tax_amt	NUMERIC,
	an_laundry_amt	NUMERIC,
	an_medicine_amt	NUMERIC,
	an_tips_amt	NUMERIC,
	an_telecommunications_amt	NUMERIC,
	an_telecommunications_csc_cscti_amt	NUMERIC,
	an_telephone_amt	NUMERIC,
	an_telephone_femsa_log_amt	NUMERIC,
	an_other_expenses_amt	NUMERIC,
	an_mileage_reimbursement_amt	NUMERIC,
	an_car_rental_vat_16_amt	NUMERIC,
	an_car_rental_vat_11_amt	NUMERIC,
	an_airport_taxi_amt	NUMERIC,
	an_taxi_amt	NUMERIC,
	an_land_transportation_amt	NUMERIC,
	an_debtor_account_amt	NUMERIC,
	an_tenure_amt	NUMERIC,
	an_other_international_expenses_amt	NUMERIC,
	an_other_international_expenses_by_employee_amt	NUMERIC,
	an_hospitality_expenses_amt	NUMERIC,
	an_not_used_flight_tickets_amt	NUMERIC,
	an_food_guests_amt	NUMERIC,
	an_other_expenses_by_company_amt	NUMERIC,
	an_other_expenses_with_receipt_amt	NUMERIC,
	an_total_cost_tax_25_amt	NUMERIC,
	an_total_cost_invoice_tax_amt	NUMERIC,
	an_nd_itesm_total_cost	NUMERIC,
	an_subtotal_amt	NUMERIC,
	an_nd_service_amt	NUMERIC,
	an_house_leasing_amt	NUMERIC,
	an_payroll_discount_amt	NUMERIC,
	an_pcr_test_amt	NUMERIC,
	an_other_federal_taxes_amt	NUMERIC,
	an_flight_fuel_amt	NUMERIC,
	an_air_navigation_service_amt	NUMERIC,
	an_landing_amt	NUMERIC,
	an_platform_parking_amt	NUMERIC,
	an_commissariat_amt	NUMERIC,
	an_national_commissariat_amt	NUMERIC,
	an_taxes_amt	NUMERIC,
	an_nd_flight_tickets_amt	NUMERIC,
	an_nd_flight_ticket_by_passenger_amt	NUMERIC,
	an_nd_international_accommodation_amt	NUMERIC,
	an_nd_food_expense_amt	NUMERIC,
	an_nd_business_lunch_expense_amt	NUMERIC,
	an_nd_fuel_amt	NUMERIC,
	an_nd_tolls_amt	NUMERIC,
	an_nd_accommodation_amt	NUMERIC,
	an_nd_telephone_amt	NUMERIC,
	an_nd_tips_amt	NUMERIC,
	an_nd_taxi_amt	NUMERIC,
	an_nd_car_rental_amt	NUMERIC,
	an_nd_parking_amt	NUMERIC,
	an_nd_land_transportation_amt	NUMERIC,
	an_nd_laundry_amt	NUMERIC,
	an_nd_medicine_amt	NUMERIC,
	an_nd_other_expenses_amt	NUMERIC,
	an_nd_food_guests_amt	NUMERIC,
	an_total_deductible_expense_usd_amt	NUMERIC,
	an_total_expense_usd_amt	NUMERIC,
	an_food_international_expense_amt	NUMERIC,
	an_food_international_expense_add_amt	NUMERIC,
	an_international_tolls_amt	NUMERIC,
	an_international_business_lunch_expense_amt	NUMERIC,
	an_international_parking_amt	NUMERIC,
	an_international_laundry_amt	NUMERIC,
	an_international_medicine_amt	NUMERIC,
	an_international_mileage_reimbursement_amt	NUMERIC,
	an_international_accommodation_amt	NUMERIC,
	an_international_car_rental_amt	NUMERIC,
	an_international_land_transportation_amt	NUMERIC,
	an_international_telecommunication_expenses_amt	NUMERIC,
	an_international_flight_tickets_amt	NUMERIC,
	an_international_hospitality_expenses_amt	NUMERIC,
	an_international_tips_amt	NUMERIC,
	an_international_taxi_amt	NUMERIC,
	an_international_fuel_amt	NUMERIC,
	an_international_flight_fuel_amt	NUMERIC,
	an_nd_international_business_lunch_expense_amt	NUMERIC,
	an_nd_int_car_rental_amt	NUMERIC,
	an_nd_int_food_expense_amt	NUMERIC,
	ax_variance_reason VARCHAR,
	PRIMARY KEY (kh_knowledge, ct_valid_from_dt)
) ;

DROP TABLE IF EXISTS multiagent_rag_model.sat_trips_adjustments;
CREATE TABLE multiagent_rag_model.sat_trips_adjustments
(
    kh_knowledge        BYTEA  NOT NULL ,
	ct_valid_from_dt     TIMESTAMP  NOT NULL ,
	ai_current_flag      INTEGER  NOT NULL ,
	ax_sub_sequence      VARCHAR  NOT NULL,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ah_checksum          BYTEA  NOT NULL ,
	ai_trip_request_id INTEGER,
	an_flight_ticket_by_passenger_adjustment_amt	NUMERIC,
	an_tua_by_passenger_adjustment_amt	NUMERIC,
	an_food_adjustments_amt	NUMERIC,
	an_nd_food_adjustments_amt	NUMERIC,
	an_nd_business_lunch_adjustment_amt	NUMERIC,
	an_business_lunch_with_receipt_adjustment_amt	NUMERIC,
	an_tolls_adjustment_amt	NUMERIC,
	an_parking_adjustment_amt	NUMERIC,
	an_fuel_adjustment_amt	NUMERIC,
	an_nd_fuel_adjustment_amt	NUMERIC,
	an_accommodation_adjustment_amt	NUMERIC,
	an_nd_accommodation_adjustment_amt	NUMERIC,
	an_accommodation_tax_adjustment_amt	NUMERIC,
	an_laundry_adjustment_amt	NUMERIC,
	an_medicine_adjustment_amt	NUMERIC,
	an_nd_other_expenses_amt	NUMERIC,
	an_other_expenses_adjustment_amt	NUMERIC,
	an_food_tips_adjustment_amt	NUMERIC,
	an_nd_tips_adjustment_amt	NUMERIC,
	an_business_lunch_tips_adjustment_amt	NUMERIC,
	an_car_rental_16_adjustment_amt	NUMERIC,
	an_car_rental_11_adjustment_amt	NUMERIC,
	an_international_car_rental_adjustment_amt	NUMERIC,
	an_telephone_with_invoice_adjustment_amt	NUMERIC,
	an_taxi_adjustment_amt	NUMERIC,
	an_land_transportation_adjustment_amt	NUMERIC,
	an_nd_land_transportation_adjustment_amt	NUMERIC,
	an_nd_taxi_adjustment_amt	NUMERIC,
	an_commission_by_passenger_adjustment_amt	NUMERIC,
	an_air_service_adjustment_amt	NUMERIC,
	an_landing_adjustment_amt	NUMERIC,
	an_platform_parking_adjustment_amt	NUMERIC,
	an_commissariat_adjustment_amt	NUMERIC,
	an_national_commissariat_adjustment_amt	NUMERIC,
	an_taxes_adjustment_amt	NUMERIC,
	an_remaining_advance_cm_amt	NUMERIC,
	an_remaining_advance_commerce_amt	NUMERIC,
	an_remaining_advance_csc_amt	NUMERIC,
	an_remaining_advance_csti_amt	NUMERIC,
	an_remaining_advance_solistica_amt	NUMERIC,
	an_remaining_advance_fs_amt	NUMERIC,
	an_remaining_advance_kof	NUMERIC,
	an_remaining_advance_cscp	NUMERIC,
	an_remaining_advance_packaging_amt	NUMERIC,
	an_remaining_advance_fs_corp_amt	NUMERIC,
	an_remaining_advance_fomento_amt	NUMERIC,
	an_remaining_advance_difusion_amt	NUMERIC,
	an_remaining_advance_fundacion_amt	NUMERIC,
	an_international_accommodation_adjustment_amt	NUMERIC,
	an_nd_international_accommodation_adjustment_amt	NUMERIC,
	an_nd_international_car_rental_amt	NUMERIC,
	an_international_food_adjustment_amt	NUMERIC,
	an_nd_international_food_adjustment_amt	NUMERIC,
	an_international_business_lunch_adjustment_amt	NUMERIC,
	an_nd_international_business_lunch_adjusment_amt	NUMERIC,
	an_international_taxi_adjustment_amt	NUMERIC,
	an_other_international_expenses_adjustment_amt	NUMERIC,
	an_international_fuel_adjustment_amt	NUMERIC,
	an_international_parking_adjustment_amt	NUMERIC,
	an_international_tolls_adjustment_amt	NUMERIC,
	an_international_flight_fuel_adjustment_amt	NUMERIC,
	an_international_flight_fuel_na_adjustment_amt	NUMERIC,
	PRIMARY KEY (kh_knowledge, ct_valid_from_dt)
) ;

DROP TABLE IF EXISTS multiagent_rag_model.sat_users_data;
CREATE TABLE multiagent_rag_model.sat_users_data
(
	kh_user            BYTEA  NOT NULL ,
	ct_valid_from_dt     TIMESTAMP  NOT NULL ,
	ai_current_flag      INTEGER  NOT NULL ,
	ct_ingest_dt         TIMESTAMP  NOT NULL ,
	ax_src_system_datastore VARCHAR  NULL ,
	ah_checksum          BYTEA  NOT NULL ,
	ax_display_nm        VARCHAR  NULL ,
	ax_user_context      VARCHAR  NULL ,
	ax_area              VARCHAR  NULL ,
	ax_email             VARCHAR  NULL,
	PRIMARY KEY (kh_user, ct_valid_from_dt)
) ;

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

DROP TABLE IF EXISTS multiagent_rag_model.sat_bv_trips_detail;
CREATE TABLE multiagent_rag_model.sat_bv_trips_detail
(
	kh_knowledge	BYTEA NOT NULL,
	ct_valid_from_dt	TIMESTAMP NOT NULL,
	ai_current_flag	INTEGER NOT NULL,
	ax_sub_sequence	VARCHAR NOT NULL,
	ct_ingest_dt	TIMESTAMP NOT NULL,
	ax_src_system_datastore	VARCHAR NULL,
	ah_checksum	BYTEA NOT NULL,
	ai_trip_request_id	integer NULL,
	ai_employee_id	integer NULL,
	ax_employee_position	VARCHAR NULL,
	ax_legal_entity_nm	VARCHAR NULL,
	ai_trip_cd	integer NULL,
	ax_direction	VARCHAR NULL,
	ax_division_id	VARCHAR NULL,
	ax_division_nm	VARCHAR NULL,
	ax_legal_entity_id	VARCHAR NULL,
	ax_hr_cost_center	VARCHAR NULL,
	ax_trip_subject	VARCHAR NULL,
	ax_trip_justification	VARCHAR NULL,
	cd_start_dt	DATE NULL,
	cd_end_dt	DATE NULL,
	cd_registry_dt	DATE NULL,
	cd_submission_dt	DATE NULL,
	cd_first_policy_validation_dt	DATE NULL,
	cd_doc_incidence_dt	TIMESTAMP NULL,
	ct_second_policy_validation_dt	TIMESTAMP NULL,
	ct_policy_incidence_dt	TIMESTAMP NULL,
	ct_final_validation_dt	TIMESTAMP NULL,
	cd_application_closing_dt	DATE NULL,
	ai_trip_days	INTEGER NULL,
	an_advance_amt	NUMERIC NULL,
	ax_trip_status	VARCHAR NULL,
	ax_origin_city_nm	VARCHAR NULL,
	ax_destination_city_nm	VARCHAR NULL,
	an_total_deductible_expense_amt	NUMERIC NULL,
	an_total_deductible_expense_usd_amt	NUMERIC NULL,
	an_total_non_deductible_expense_amt	NUMERIC NULL,
	an_total_non_deductible_expense_usd_amt	NUMERIC NULL,
	an_total_expense_amt	NUMERIC NULL,
	an_total_expense_usd_amt	NUMERIC NULL,
	an_other_international_expenses_amt	NUMERIC NULL,
	an_food_expense_amt	NUMERIC NULL,
	an_food_international_expense_amt	NUMERIC NULL,
	an_tolls_amt	NUMERIC NULL,
	an_flight_tickets_amt	NUMERIC NULL,
	an_int_tips_amt	NUMERIC NULL,
	an_flight_tickets_agency_commission_by_company_amt	NUMERIC NULL,
	an_commission_by_company_amt	NUMERIC NULL,
	an_parking_amt	NUMERIC NULL,
	an_fuel_amt	NUMERIC NULL,
	an_accommodation_amt	NUMERIC NULL,
	an_international_laundry_amt	NUMERIC NULL,
	an_international_medicine_amt	NUMERIC NULL,
	an_laundry_amt	NUMERIC NULL,
	an_medicine_amt	NUMERIC NULL,
	an_other_expenses_amt	NUMERIC NULL,
	an_tips_amt	NUMERIC NULL,
	an_car_rental_vat_16_amt	NUMERIC NULL,
	an_taxi_amt	NUMERIC NULL,
	an_airport_taxi_amt	NUMERIC NULL,
	an_land_transportation_amt	NUMERIC NULL,
	an_nd_food_expense_amt	NUMERIC NULL,
	an_nd_business_lunch_expense_amt	NUMERIC NULL,
	an_nd_fuel_amt	NUMERIC NULL,
	an_nd_accommodation_amt	NUMERIC NULL,
	an_nd_other_expenses_amt	NUMERIC NULL,
	an_nd_tips_amt	NUMERIC NULL,
	an_nd_taxi_amt	NUMERIC NULL,
	an_nd_land_transportation_amt	NUMERIC NULL,
	an_flight_ticket_by_passenger_amt	NUMERIC NULL,
	an_flight_tickets_agency_commission_by_passenger_amt	NUMERIC NULL,
	an_remaining_advance_commerce_amt	NUMERIC NULL,
	an_international_car_rental_amt	NUMERIC NULL,
	an_car_rental_vat_11_amt	NUMERIC NULL,
	an_international_accommodation_amt	NUMERIC NULL,
	an_telephone_amt	NUMERIC NULL,
	an_international_land_transportation_amt	NUMERIC NULL,
	an_international_parking_amt	NUMERIC NULL,
	an_nd_laundry_amt	NUMERIC NULL,
	an_nd_flight_tickets_amt	NUMERIC NULL,
	an_nd_tolls_amt	NUMERIC NULL,
	an_nd_telephone_amt	NUMERIC NULL,
	an_nd_parking_amt	NUMERIC NULL,
	an_int_taxi_amt	NUMERIC NULL,
	an_int_fuel_amt	NUMERIC NULL,
	an_international_tolls_amt	NUMERIC NULL,
	an_nd_flight_ticket_by_passenger_amt	NUMERIC NULL,
	an_nd_international_accommodation_amt	NUMERIC NULL,
	an_nd_int_car_rental_amt	NUMERIC NULL,
	an_nd_int_food_expense_amt	NUMERIC NULL,
	an_international_business_lunch_expense_amt	NUMERIC NULL,
	an_nd_international_business_lunch_expense_amt	NUMERIC NULL,
	an_food_international_expense_add_amt	NUMERIC NULL,
	an_int_flight_tickets_amt	NUMERIC NULL,
	an_flight_ticket_by_company_amt	NUMERIC NULL,
	an_nd_car_rental_amt	NUMERIC NULL,
	an_other_expenses_by_company_amt	NUMERIC NULL,
	an_flight_ticket_by_passenger_cash_card_amt	NUMERIC NULL,
	ab_trip_by_air	BOOLEAN NULL,
	ab_trip_by_land	BOOLEAN NULL,
	ax_trip_type	NUMERIC NULL,
	ax_air_transport_type	VARCHAR NULL,
	ax_land_transport_type	VARCHAR ,
	ax_approver_nm VARCHAR NULL,
	ax_variance_reason VARCHAR NULL,
	PRIMARY KEY (kh_knowledge, ct_valid_from_dt)
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

CREATE TABLE multiagent_rag_model.bcd_travel_avion (
    id integer NOT NULL,
    client_name text,
    record_key text,
    global_customer_number bigint,
    client_code bigint,
    locator text,
    traveler text,
    invoice_date timestamp without time zone,
    invoice_number bigint,
    agency_name text,
    agency_code text,
    booking_source text,
    booking_agent_id double precision,
    local_air_reason_code text,
    local_air_reason_code_description text,
    global_air_reason_code text,
    global_air_reason_code_description text,
    fare_accepted_code text,
    fare_accepted_code_description text,
    credit_card_number text,
    credit_card_type text,
    credit_card_expiration double precision,
    refund_indicator text,
    exchange_indicator text,
    true_ticket_count bigint,
    number_of_tickets bigint,
    number_of_refunds bigint,
    number_of_exchanges bigint,
    round_trip_indicator text,
    short_long_haul_indicator text,
    original_document_number bigint,
    int_dom text,
    travel_sector text,
    traveler_country text,
    ticketing_country text,
    traveler_region text,
    ticketing_region text,
    trip_length bigint,
    travel_start_date timestamp without time zone,
    ticket_number bigint,
    carrier_code text,
    carrier_name text,
    origin_airport_code text,
    origin_city text,
    origin_airport text,
    origin_country text,
    destination_airport_code text,
    destination_city text,
    destination_airport text,
    destination_country text,
    routing text,
    booking_class_summary text,
    fare_basis_summary text,
    cabin text,
    tour_code text,
    ticket_designator text,
    booking_date timestamp without time zone,
    days_advance_booking bigint,
    days_advance_booking_group text,
    days_advance_purchase bigint,
    days_advance_purchase_group text,
    mileage double precision,
    kilometers double precision,
    _cost_per_mile_mxn double precision,
    tax_amount_mxn double precision,
    ticket_amt_minus_taxes_mxn double precision,
    total_ticket_amount_mxn double precision,
    low_fare_mxn double precision,
    full_fare_mxn double precision,
    amount_lost_mxn double precision,
    full_fare_savings_mxn double precision,
    contract_savings_mxn double precision,
    fare_before_discount_mxn double precision,
    fare_compare_2_mxn bigint,
    fare_compare_3_mxn double precision,
    fare_compare_4_mxn bigint,
    fare_compare_5_mxn bigint,
    fare_compare_6_mxn bigint,
    vat_on_fees double precision,
    source_currency_code text,
    source_currency_tax_amount double precision,
    source_currency_ticket_amt_minus_taxes double precision,
    source_currency_total_ticket_amount double precision,
    source_currency_low_fare double precision,
    source_currency_full_fare double precision,
    source_currency_fare_before_discount double precision,
    update_date timestamp without time zone,
    original_import_date timestamp without time zone,
    traveler_email_address text,
    regional_indicator text,
    destination_region text,
    origin_region text,
    return_date timestamp without time zone,
    round_trip_description text,
    defra_kg_co2e double precision,
    defra_tonnes_co2e double precision,
    defra_incl__wtt_kg_co2e double precision,
    defra_incl__wtt_tonnes_co2e double precision,
    defra_with_rfi_kg_co2e double precision,
    defra_with_rfi_tonnes_co2e double precision,
    defra_with_rfi_incl__wtt_kg_co2e double precision,
    defra_with_rfi_incl__wtt_tonnes_co2e double precision,
    epa_kg_co2e double precision,
    epa_tonnes_co2e double precision,
    cost_center text,
    employee_number bigint,
    employee_type text,
    no_hotel_booked double precision,
    purpose_of_trip double precision,
    rmk048 bigint,
    rmk076 text,
    rmk078 bigint,
    rmk108 text,
    rmk134 double precision,
    rmk141 text,
    rmk143 text,
    rmk144 text,
    rmk147 double precision,
    rmk150 text,
    rmk155 text,
    rmk156 text,
    rmk157 text,
    rmk168 bigint,
    rmk169 double precision,
    rmk170 double precision,
    rmk183 text,
    sap_authorization text,
    zz_rmk009 text,
    zz_rmk047 double precision,
    zz_rmk051 bigint,
    zz_rmk057 double precision,
    zz_rmk058 double precision,
    zz_rmk059 double precision,
    zz_rmk063 double precision,
    zz_rmk103 double precision,
    zz_rmk105 double precision,
    zz_rmk131 double precision,
    zz_rmk133 bigint
);

CREATE SEQUENCE multiagent_rag_model.bcd_travel_avion_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

ALTER TABLE ONLY multiagent_rag_model.bcd_travel_avion
    ADD CONSTRAINT bcd_travel_avion_pkey PRIMARY KEY (id);

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
