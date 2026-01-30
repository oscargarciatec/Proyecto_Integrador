# Guía de Despliegue Local y Ngrok para Compass

## 1. Prerrequisitos
- **Docker** y **Docker Compose** (opcional, recomendado)
- **Python 3.11+** (si no usas Docker)
- **Ngrok** (para exponer tu local a Slack)
- Una **Slack App** creada en [api.slack.com/apps](https://api.slack.com/apps)

## 2. Configuración de Entorno
1. Copia el archivo de ejemplo:
   ```bash
   cp .env.example .env
   ```
2. Edita `.env` y añade tus credenciales. 
   - Para Slack, necesitas el `SLACK_BOT_TOKEN` y `SLACK_SIGNING_SECRET`.
   - Para GCP/VertexAI, asegúrate de tener las credenciales por defecto de gcloud o un Service Account.

## 3. Ejecutar la Aplicación

### Opción A: Docker (Recomendada)
Desde la carpeta `src/compass`:
```bash
docker-compose up --build
```
La app correrá en `http://localhost:8080`.

### Opción B: Python Local
```bash
pip install -r requirements.txt
uvicorn app_async:api --host 0.0.0.0 --port 8080 --reload
```

## 4. Configurar Ngrok (Túnel a Internet)
Slack necesita una URL pública para enviar eventos. Ngrok crea un túnel seguro desde internet a tu puerto 8080.

### Paso 4.1: Instalación
Si estás en Mac:
```bash
brew install ngrok/ngrok/ngrok
```
(O descarga desde [ngrok.com](https://ngrok.com/download))

### Paso 4.2: Autenticación
Debes tener una cuenta en Ngrok. Ve al dashboard, copia tu token y ejecuta:
```bash
ngrok config add-authtoken TU_TOKEN_AQUI
```

### Paso 4.3: Iniciar el Túnel
En una **nueva terminal**:
```bash
ngrok http 8080
```
Verás una salida como:
`Forwarding https://abcd-1234.ngrok-free.app -> http://localhost:8080`

Copia la URL `https` (ej: `https://abcd-1234.ngrok-free.app`).

## 5. Configurar Slack App
1. Ve a "Event Subscriptions" en tu Slack App dashboard.
2. Activa "Enable Events".
3. En "Request URL", pega tu URL de ngrok seguida de `/slack/events`:
   `https://abcd-1234.ngrok-free.app/slack/events`
4. Debería aparecer "Verified" ✅.
5. Suscríbete a los eventos necesarios (ej: `message.channels`, `app_mention`).
6. Reinstala la App en tu Workspace si es necesario.
