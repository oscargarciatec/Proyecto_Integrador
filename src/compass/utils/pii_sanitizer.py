"""
Sanitizador de PII (Personally Identifiable Information).
Tokeniza/enmascara PII en el input antes de enviarlo al LLM.
"""

import re
from dataclasses import dataclass, field


@dataclass
class SanitizationResult:
    """Resultado de la sanitización de un texto."""

    sanitized_text: str
    tokens: dict[str, str] = field(default_factory=dict)  # {token: valor_original}
    pii_found: list[str] = field(default_factory=list)  # tipos de PII encontrados

    @property
    def has_pii(self) -> bool:
        return len(self.tokens) > 0


class PIISanitizer:
    """
    Sanitiza PII reemplazándolo por tokens seguros.
    Los tokens permiten tracking sin exponer datos reales.
    """

    # Patrones de PII con grupos de captura
    # IMPORTANTE: El orden importa - tarjetas y CLABE van ANTES que teléfonos
    # para evitar que números de 15-18 dígitos se detecten como teléfonos
    PII_PATTERNS = {
        "email": {
            "regex": r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b",
            "token": "[EMAIL_REDACTED_{n}]",
            "new_token": "<PII_IDENTIFICATION>",
        },
        "curp": {
            "regex": r"\b([A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]\d)\b",
            "token": "[CURP_REDACTED_{n}]",
            "new_token": "<PII_IDENTIFICATION>",
        },
        "rfc": {
            "regex": r"\b([A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3})\b",
            "token": "[RFC_REDACTED_{n}]",
            "new_token": "<PII_FINANCIAL>",
        },
        # Tarjetas de crédito: 16 dígitos (Visa/MC) o 15 dígitos (AMEX)
        # Formatos: con guiones, espacios, o sin separadores
        "credit_card": {
            "regex": (
                r"\b("
                r"(?:\d{4}[\W_]{0,2}){3}\d{4}"  # 16 dígitos: XXXX-XXXX-XXXX-XXXX
                r"|\d{4}[\W_]{0,2}\d{6}[\W_]{0,2}\d{5}"  # AMEX 15 dígitos: XXXX-XXXXXX-XXXXX
                r"|\d{15,16}"  # 15-16 dígitos sin separadores
                r")\b"
            ),
            "token": "[CARD_REDACTED_{n}]",
            "new_token": "<PII_FINANCIAL>",
        },
        "clabe": {
            "regex": r"\b(\d{18})\b",
            "token": "[CLABE_REDACTED_{n}]",
            "new_token": "<PII_FINANCIAL>",
        },
        "account_number": {
            "regex": r"\b(\d{10,12})\b",
            "token": "[ACCOUNT_REDACTED_{n}]",
            "new_token": "<PII_FINANCIAL>",
        },
        # Teléfonos mexicanos - múltiples formatos comunes
        # Nota: Va después de tarjetas para evitar falsos positivos
        "phone": {
            "regex": (
                r"\b("
                r"(?:\+?52\s?1?\s?)?"  # Prefijo internacional opcional (+52 o 52, con 1 opcional)
                r"(?:"
                # Formato 1: Lada (2-3 dígitos) + 7-8 dígitos locales (ej: 55 1234 5678, 442-123-4567)
                r"(?:\(?\d{2,3}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}"
                r"|"
                # Formato 2: XX XX XXXXXX (ej: 55 56 104531) - común en México
                r"\d{2}[-.\s]\d{2}[-.\s]\d{6}"
                r"|"
                # Formato 3: 10 dígitos seguidos (ej: 5512345678)
                r"\d{10}"
                r")"
                r")\b"
            ),
            "token": "[PHONE_REDACTED_{n}]",
            "new_token": "<PII_IDENTIFICATION>",
        },
        "ine": {
            "regex": r"\b(\d{13})\b",
            "token": "[INE_REDACTED_{n}]",
            "new_token": "<PII_IDENTIFICATION>",
        },
    }

    # Patrones de montos (para no loguear cantidades específicas de salario)
    AMOUNT_PATTERNS = {
        "salary_amount": {
            "regex": r"(?:gano|cobro|salario|sueldo|ingreso|bono|aguinaldo)(?:[^\d\n]{0,40}?)\$?([\d,]*\d[\d,]*(?:\.\d{2})?)",
            "token": "[AMOUNT_REDACTED_{n}]",
            "new_token": "<PII_SOCIAL>",
        },
    }

    @classmethod
    def sanitize(
        cls, text: str, redact_amounts: bool = False, new_schema: bool = False
    ) -> SanitizationResult:
        """
        Sanitiza el texto reemplazando PII por tokens.

        Args:
            text: Texto a sanitizar
            redact_amounts: Si True, también redacta montos de salario

        Returns:
            SanitizationResult con texto sanitizado y mapeo de tokens
        """
        if not text:
            return SanitizationResult(sanitized_text="", tokens={}, pii_found=[])

        sanitized = text
        tokens: dict[str, str] = {}
        pii_found: list[str] = []
        counter = 0

        patterns_dict = cls.PII_PATTERNS
        if redact_amounts:
            patterns_dict = cls.PII_PATTERNS | cls.AMOUNT_PATTERNS

        # Procesar patrones de PII y Montos
        for pii_type, pii_dict in patterns_dict.items():
            matches = list(re.finditer(pii_dict["regex"], sanitized, re.IGNORECASE))
            for match in reversed(matches):  # Reversed para no afectar índices
                original_value = match.group(1)
                counter += 1
                token = pii_dict["new_token"] if new_schema else pii_dict["token"]
                if not new_schema:
                    token = token.format(n=counter)
                tokens[token] = original_value
                pii_found.append(pii_type)
                sanitized = (
                    sanitized[: match.start(1)] + token + sanitized[match.end(1) :]
                )

        return SanitizationResult(
            sanitized_text=sanitized, tokens=tokens, pii_found=pii_found
        )

    @classmethod
    def get_safe_log_text(cls, text: str) -> str:
        """
        Retorna una versión del texto segura para logging/persistencia.
        Usa tokens genéricos por categoría (<PII_FINANCIAL>, etc.) para
        mantener consistencia con el texto enviado al LLM.
        No guarda el mapeo de tokens (one-way).
        """
        result = cls.sanitize(text, redact_amounts=True, new_schema=True)
        return result.sanitized_text

    @classmethod
    def extract_pii_types(cls, text: str) -> list[str]:
        """Retorna lista de tipos de PII encontrados en el texto."""
        result = cls.sanitize(text)
        return list(set(result.pii_found))
