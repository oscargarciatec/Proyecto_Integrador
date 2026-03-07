import os
import tempfile
import logging

logger = logging.getLogger(__name__)

def setup_gcp_credentials():
    """
    Sets up GCP credentials. 
    If GCP_SERVICE_ACCOUNT_JSON is present in the environment, it writes it to a 
    temporary file and sets GOOGLE_APPLICATION_CREDENTIALS to that file path.
    This allows deployment on platforms like Render without uploading sensitive JSON files.
    """
    credentials_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    
    if credentials_json:
        try:
            # Handle potential escaping issues if the string was added via some UI
            if credentials_json.startswith("'") and credentials_json.endswith("'"):
                credentials_json = credentials_json[1:-1]
            
            # Create a temporary file that persists while the process is running
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
                temp_file.write(credentials_json)
                temp_path = temp_file.name
            
            # Set the environment variable that Google SDKs expect
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
            logger.info(f"GCP Credentials configured from GCP_SERVICE_ACCOUNT_JSON (Path: {temp_path})")
        except Exception as e:
            logger.error(f"Failed to setup GCP credentials from environment variable: {e}")
    else:
        # Fallback to existing GOOGLE_APPLICATION_CREDENTIALS or default auth
        current_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if current_creds:
            logger.info(f"Using existing GOOGLE_APPLICATION_CREDENTIALS file: {current_creds}")
        else:
            logger.warning("No GCP credentials found (GCP_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS)")
