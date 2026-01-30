# utils/rate_limiter.py
"""
Rate limiter para proteger el servicio contra abuso.
Limita requests por usuario usando ventana deslizante en memoria.
"""
import time
from collections import defaultdict
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter en memoria con ventana deslizante.
    Thread-safe para uso con asyncio (no usa locks porque es single-threaded).
    """

    def __init__(
        self,
        max_requests: int = 20,
        window_seconds: int = 60,
        burst_limit: int = 5,
        burst_window_seconds: int = 10,
    ):
        """
        Args:
            max_requests: Máximo de requests permitidos en la ventana principal
            window_seconds: Tamaño de la ventana principal en segundos
            burst_limit: Máximo de requests en ráfaga (ventana corta)
            burst_window_seconds: Tamaño de la ventana de ráfaga
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.burst_limit = burst_limit
        self.burst_window_seconds = burst_window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def _cleanup_old_requests(self, user_id: str, now: float) -> None:
        """Elimina requests antiguos fuera de la ventana."""
        cutoff = now - self.window_seconds
        self._requests[user_id] = [
            t for t in self._requests[user_id] if t > cutoff
        ]

    def is_allowed(self, user_id: str) -> Tuple[bool, str]:
        """
        Verifica si el usuario puede hacer un request.
        
        Returns:
            Tuple[bool, str]: (permitido, mensaje de error si no permitido)
        """
        now = time.time()
        self._cleanup_old_requests(user_id, now)

        requests = self._requests[user_id]

        # Verificar límite de ráfaga (ventana corta)
        burst_cutoff = now - self.burst_window_seconds
        recent_burst = sum(1 for t in requests if t > burst_cutoff)
        if recent_burst >= self.burst_limit:
            wait_time = int(self.burst_window_seconds - (now - min(t for t in requests if t > burst_cutoff)))
            logger.warning(f"Rate limit (burst) exceeded for user {user_id}")
            return False, f"Demasiados mensajes seguidos. Espera {wait_time} segundos."

        # Verificar límite principal
        if len(requests) >= self.max_requests:
            oldest = min(requests)
            wait_time = int(self.window_seconds - (now - oldest))
            logger.warning(f"Rate limit exceeded for user {user_id}: {len(requests)} requests")
            return False, f"Has alcanzado el límite de mensajes. Espera {wait_time} segundos."

        # Permitido: registrar el request
        self._requests[user_id].append(now)
        return True, ""

    def get_remaining(self, user_id: str) -> int:
        """Retorna cuántos requests le quedan al usuario."""
        now = time.time()
        self._cleanup_old_requests(user_id, now)
        return max(0, self.max_requests - len(self._requests[user_id]))

    def reset(self, user_id: str) -> None:
        """Resetea el contador de un usuario (para admins)."""
        if user_id in self._requests:
            del self._requests[user_id]

    def cleanup_all(self) -> int:
        """
        Limpia todos los registros expirados.
        Útil para llamar periódicamente y liberar memoria.
        Returns: número de usuarios limpiados
        """
        now = time.time()
        cutoff = now - self.window_seconds
        users_to_remove = []

        for user_id, requests in self._requests.items():
            # Filtrar requests antiguos
            self._requests[user_id] = [t for t in requests if t > cutoff]
            # Si no quedan requests, marcar para eliminar
            if not self._requests[user_id]:
                users_to_remove.append(user_id)

        for user_id in users_to_remove:
            del self._requests[user_id]

        return len(users_to_remove)
