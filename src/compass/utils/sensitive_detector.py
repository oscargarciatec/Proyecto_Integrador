# utils/sensitive_detector.py
"""
Detector de contenido sensible en mensajes de usuarios.
Identifica PII y temas que requieren respuesta privada.
"""

import re
from typing import Tuple


class SensitiveContentDetector:
    """
    Detecta si un mensaje contiene información sensible que
    debería responderse de forma privada (ephemeral) en canales públicos.
    """

    # Patrones de PII
    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        # Teléfonos mexicanos - múltiples formatos
        "phone": (
            r"\b(?:\+?52\s?1?\s?)?"  # Prefijo internacional opcional
            r"(?:"
            r"(?:\(?\d{2,3}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}"  # Formato estándar
            r"|"
            r"\d{2}[-.\s]\d{2}[-.\s]\d{6}"  # Formato XX XX XXXXXX
            r"|"
            r"\d{10}"  # 10 dígitos seguidos
            r")\b"
        ),
        "curp": r"\b[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]\d\b",
        "rfc": r"\b[A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3}\b",
        # Tarjetas de crédito: 16 dígitos (Visa/MC) o 15 dígitos (AMEX)
        "credit_card": (
            r"\b(?:"
            r"(?:\d{4}[-\s]?){3}\d{4}"  # 16 dígitos: XXXX-XXXX-XXXX-XXXX
            r"|\d{4}[-\s]?\d{6}[-\s]?\d{5}"  # AMEX: XXXX-XXXXXX-XXXXX
            r"|\d{15,16}"  # 15-16 dígitos sin separadores
            r")\b"
        ),
        "clabe": r"\b\d{18}\b",  # CLABE interbancaria
        "account_number": r"\b\d{10,12}\b",  # Número de cuenta bancaria
        "ine": r"\b\d{13}\b",
    }

    # Palabras clave que indican información personal/financiera
    SENSITIVE_KEYWORDS = [
        # Financiero personal
        r"\b(?:mi\s+)?salario\b",
        r"\b(?:mi\s+)?sueldo\b",
        r"\bgano\b",
        r"\bcobro\b",
        r"\b(?:mi\s+)?compensaci[oó]n\b",
        r"\b(?:mi\s+)?bono\b",
        r"\b(?:mi\s+)?aguinaldo\b",
        # Datos personales
        r"\b(?:mi\s+)?n[uú]mero\s+de\s+empleado\b",
        r"\b(?:mi\s+)?cuenta\s+bancaria\b",
        r"\b(?:mi\s+)?clabe\b",
        r"\b(?:mi\s+)?tarjeta\b",
        # Situaciones personales
        r"\b(?:mi\s+)?incapacidad\b",
        r"\b(?:mi\s+)?licencia\s+m[eé]dica\b",
        r"\b(?:mi\s+)?embarazo\b",
        r"\b(?:mi\s+)?enfermedad\b",
        r"\bproblema\s+personal\b",
        # Casos excepcionales
        r"\bexcepci[oó]n\b",
        r"\bcaso\s+especial\b",
        r"\bautorizaci[oó]n\s+especial\b",
    ]

    @classmethod
    def is_sensitive(cls, text: str) -> Tuple[bool, str]:
        """
        Detecta si el texto contiene información sensible.

        Returns:
            Tuple[bool, str]: (es_sensible, razón)
        """
        if not text:
            return False, ""

        text_lower = text.lower()

        # Verificar PII
        for pii_type, pattern in cls.PII_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return True, f"pii_{pii_type}"

        # Verificar palabras clave sensibles
        for pattern in cls.SENSITIVE_KEYWORDS:
            if re.search(pattern, text_lower):
                return True, "sensitive_topic"

        return False, ""

    @classmethod
    def get_ephemeral_warning(cls) -> str:
        """Mensaje de advertencia para respuestas ephemeral."""
        return (
            "🔒 *Tu mensaje parece contener información personal o sensible.*\n\n"
            "Por tu privacidad, te responderé por mensaje directo. "
            "También puedes escribirme directamente en cualquier momento."
        )

    @classmethod
    def get_dm_suggestion(cls) -> str:
        """Sugerencia para ir a DM."""
        return (
            "💡 *Tip:* Para consultas que involucren información personal "
            "(salarios, casos especiales, etc.), te recomiendo escribirme "
            "por mensaje directo para mayor privacidad."
        )
