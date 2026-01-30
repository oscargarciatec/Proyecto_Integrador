# utils/output_validator.py
"""
Validador de output del LLM.
Verifica que la respuesta no contenga PII que el usuario haya mencionado.
"""
import re
from typing import List
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Resultado de la validación de output."""
    is_safe: bool
    violations: List[str]  # Lista de PII encontrado en el output
    sanitized_output: str  # Output con PII redactado (si había violaciones)


class OutputValidator:
    """
    Valida que el output del LLM no repita PII del usuario.
    Actúa como última línea de defensa antes de enviar la respuesta.
    """

    # Patrones de PII que NO deberían aparecer en respuestas
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+52\s?)?(?:\d{2,3}[-.\s]?)?\d{4}[-.\s]?\d{4}\b',
        'curp': r'\b[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]\d\b',
        'rfc': r'\b[A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3}\b',
        'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        'clabe': r'\b\d{18}\b',
    }

    # Patrones que indican que el LLM está repitiendo info del usuario
    ECHO_PATTERNS = [
        r'(?:tu|su)\s+(?:RFC|CURP|email|correo|teléfono|tarjeta)\s+(?:es|fue|era)\s*:?\s*\S+',
        r'(?:mencionaste|dijiste|indicaste)\s+(?:que\s+)?(?:tu|su)',
    ]

    @classmethod
    def validate(
        cls,
        output: str,
        user_input: str = None,
        extracted_pii: List[str] = None
    ) -> ValidationResult:
        """
        Valida que el output no contenga PII.
        
        Args:
            output: Respuesta del LLM a validar
            user_input: Input original del usuario (para detectar ecos)
            extracted_pii: Lista de valores PII extraídos del input (si ya se sanitizó)
            
        Returns:
            ValidationResult con estado de validación
        """
        if not output:
            return ValidationResult(is_safe=True, violations=[], sanitized_output="")

        violations: List[str] = []
        sanitized = output

        # 1. Buscar PII genérico en el output
        for pii_type, pattern in cls.PII_PATTERNS.items():
            matches = re.findall(pattern, output, re.IGNORECASE)
            for match in matches:
                # Verificar si este PII estaba en el input del usuario
                if user_input and match.lower() in user_input.lower():
                    violations.append(f"{pii_type}:{match}")
                    # Redactar en el output sanitizado
                    sanitized = sanitized.replace(match, f"[{pii_type.upper()}_REMOVED]")

        # 2. Si tenemos PII extraído del input, verificar que no aparezca
        if extracted_pii:
            for pii_value in extracted_pii:
                if pii_value.lower() in output.lower():
                    violations.append(f"echoed_pii:{pii_value[:4]}***")
                    sanitized = re.sub(
                        re.escape(pii_value),
                        "[PII_REMOVED]",
                        sanitized,
                        flags=re.IGNORECASE
                    )

        # 3. Detectar patrones de eco (LLM repitiendo lo que dijo el usuario)
        for pattern in cls.ECHO_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                violations.append("echo_pattern_detected")
                break

        return ValidationResult(
            is_safe=len(violations) == 0,
            violations=violations,
            sanitized_output=sanitized if violations else output
        )

    @classmethod
    def validate_and_fix(
        cls,
        output: str,
        user_input: str = None,
        extracted_pii: List[str] = None
    ) -> str:
        """
        Valida y retorna output seguro.
        Si hay violaciones, retorna la versión sanitizada.
        """
        result = cls.validate(output, user_input, extracted_pii)
        return result.sanitized_output

    @classmethod
    def get_pii_warning_suffix(cls) -> str:
        """Mensaje a añadir si se detectó y removió PII."""
        return (
            "\n\n_Nota: Por seguridad, algunos datos personales fueron "
            "omitidos de esta respuesta._"
        )
