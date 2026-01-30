# config/settings.py
import os
import sys
from configparser import ConfigParser
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()


@dataclass(frozen=True)
class ChatbotSettings:
    """
    Centralized configuration management using OOP (dataclass).
    Loads from config.ini and environment variables.
    """

    # Config.ini settings
    db_region: str
    db_cluster: str
    db_instance: str
    db_name: str
    db_user: str
    k_sim_search_num: int
    chatbot_logs_path: str
    llm_model_name: str
    embedding_model_name: str
    db_schema_name: str
    vector_column_name: str
    vertex_ai_project_id: str
    vertex_ai_region: str
    embedding_column_name: str
    fts_vector_column_name: str

    # Environment Variables (Secrets/Tokens)
    slack_bot_token: str = field(
        default_factory=lambda: os.environ.get("SLACK_BOT_TOKEN")
    )
    slack_signing_secret: str = field(
        default_factory=lambda: os.environ.get("SLACK_SIGNING_SECRET")
    )
    db_password: str = field(default_factory=lambda: os.environ.get("DB_PASSWORD"))
    collection_table_name: str = field(
        default_factory=lambda: os.environ.get("KNOWLEDGE_TABLE_NAME")
    )
    gcp_project_id: str = field(
        default_factory=lambda: os.environ.get("GCP_PROJECT_ID")
    )
    db_ip_type: str = field(default_factory=lambda: os.environ.get("ALLOYDB_IP_TYPE"))
    db_host_ip: str = field(default_factory=lambda: os.environ.get("DB_HOST_IP"))

    _config_path: str = "config/config.ini"

    def __post_init__(self):
        """Validates environment variables and loads config.ini."""
        self._validate_secrets()
        self._load_ini_config()

    def _validate_secrets(self):
        """Checks for required environment variables."""
        required_vars = {
            "DB_PASSWORD": self.db_password,
            "GCP_PROJECT_ID": self.gcp_project_id,
        }

        optional_vars = {
            "SLACK_BOT_TOKEN": self.slack_bot_token,
            "SLACK_SIGNING_SECRET": self.slack_signing_secret,
            "KNOWLEDGE_TABLE_NAME": self.collection_table_name,
            "DB_HOST_IP": self.db_host_ip,
        }

        missing_required = [name for name, val in required_vars.items() if not val]
        if missing_required:
            print(
                f"Error: Missing required environment variables: {', '.join(missing_required)}"
            )
            sys.exit(1)

        missing_optional = [name for name, val in optional_vars.items() if not val]
        if missing_optional:
            print(
                f"Warning: Optional environment variables not set: {', '.join(missing_optional)}"
            )

    def _load_ini_config(self):
        """Loads non-secret settings from the INI file, allowing env var overrides."""
        config = ConfigParser()
        try:
            if not config.read(self._config_path):
                print(
                    f"Warning: Config file not found at {self._config_path}, relying on environment variables."
                )
        except Exception as e:
            print(f"Error reading config file: {e}")
            # Don't exit here, maybe env vars are enough

        # Helper to get value from env or config
        def get_conf(env_key, section, key, default=None):
            val = os.environ.get(env_key)
            if val:
                return val
            try:
                return config[section][key]
            except (KeyError, NameError):  # NameError if config not defined
                return default

        # Using setattr for immutable fields in dataclasses
        object.__setattr__(
            self, "db_region", get_conf("DB_REGION", "AlloyDB", "DB_REGION")
        )
        object.__setattr__(
            self, "db_cluster", get_conf("DB_CLUSTER", "AlloyDB", "DB_CLUSTER")
        )
        object.__setattr__(
            self, "db_instance", get_conf("DB_INSTANCE", "AlloyDB", "DB_INSTANCE")
        )
        object.__setattr__(self, "db_name", get_conf("DB_NAME", "AlloyDB", "DB_NAME"))
        object.__setattr__(self, "db_user", get_conf("DB_USER", "AlloyDB", "DB_USER"))

        k_sim = get_conf("K_SIM_SEARCH_NUM", "AlloyDB", "K_SIM_SEARCH_NUM", "10")
        object.__setattr__(self, "k_sim_search_num", int(k_sim))

        object.__setattr__(
            self,
            "db_schema_name",
            get_conf("DB_SCHEMA_NAME", "AlloyDB", "DB_SCHEMA_NAME"),
        )
        object.__setattr__(
            self,
            "vector_column_name",
            get_conf("VECTOR_COLUMN_NAME", "AlloyDB", "VECTOR_COLUMN_NAME"),
        )
        object.__setattr__(
            self,
            "embedding_column_name",
            get_conf(
                "EMBEDDING_COLUMN_NAME",
                "AlloyDB",
                "EMBEDDING_COLUMN_NAME",
                "aa_embedding",
            ),
        )
        object.__setattr__(
            self,
            "fts_vector_column_name",
            get_conf(
                "FTS_VECTOR_COLUMN_NAME",
                "AlloyDB",
                "FTS_VECTOR_COLUMN_NAME",
                "aa_ts_vector",
            ),
        )

        object.__setattr__(
            self,
            "collection_table_name",
            get_conf("KNOWLEDGE_TABLE_NAME", "AlloyDB", "KNOWLEDGE_TABLE_NAME"),
        )

        object.__setattr__(
            self,
            "llm_model_name",
            get_conf("LLM_MODEL_NAME", "Model", "llm_model_name"),
        )
        object.__setattr__(
            self,
            "embedding_model_name",
            get_conf("EMBEDDING_MODEL_NAME", "Model", "embedding_model_name"),
        )

        # Vertex AI Project
        vertex_project = os.environ.get("VERTEX_AI_PROJECT_ID")
        if not vertex_project:
            try:
                vertex_project = config["Model"].get("vertex_ai_project_id")
            except:
                pass
        if not vertex_project:
            vertex_project = self.gcp_project_id
        object.__setattr__(self, "vertex_ai_project_id", vertex_project)

        # Vertex AI Region
        vertex_region = os.environ.get("VERTEX_AI_REGION")
        if not vertex_region:
            try:
                vertex_region = config["Model"].get("vertex_ai_region", self.db_region)
            except:
                vertex_region = self.db_region
        object.__setattr__(self, "vertex_ai_region", vertex_region)

        # Logs path
        object.__setattr__(
            self,
            "chatbot_logs_path",
            get_conf("CHATBOT_LOGS", "Data", "chatbot_logs", "../data/logs/"),
        )

        # Debug logging
        print(f"--- Configuration Loaded ---")
        print(f"Project ID: {self.gcp_project_id}")
        print(f"Region: {self.db_region}")
        print(f"Cluster: {self.db_cluster}")
        print(f"Instance: {self.db_instance}")
        print(f"DB Name: {self.db_name}")
        print(f"IP Type: {self.db_ip_type}")
        print(f"DB Host IP: {self.db_host_ip}")
        print(f"Vertex Project: {self.vertex_ai_project_id}")
        print(f"----------------------------")


def load_settings() -> ChatbotSettings:
    """Factory function to load and return the settings object."""
    # Initialize with dummy values. The __post_init__ method will overwrite
    # these with actual values from config.ini and environment variables.
    return ChatbotSettings(
        chatbot_logs_path="",
        db_region="",
        db_cluster="",
        db_instance="",
        db_name="",
        db_user="",
        k_sim_search_num=0,
        llm_model_name="",
        embedding_model_name="",
        db_schema_name="",
        vector_column_name="",
        vertex_ai_project_id="",
        vertex_ai_region="",
        embedding_column_name="",
        fts_vector_column_name="",
    )
