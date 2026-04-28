"""
Math Linearizer Module

Converts mathematical notation to spoken form for blind students.
Handles edge cases and provides robust error handling.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MathLinearizer:
    """Converts mathematical notation to spoken form."""

    # Comprehensive math symbol mappings
    MATH_SYMBOLS = {
        # Powers and subscripts
        "²": "squared",
        "³": "cubed",
        "^2": " to the power of 2",
        "^3": " to the power of 3",
        "^n": " to the power of n",
        "^-1": " to the power of negative one",
        "^-2": " to the power of negative two",
        "_": " sub ",

        # Calculus symbols
        "∫": "the integral from",
        "∫∫": "the double integral of",
        "∮": "the closed integral of",
        "∑": "the summation of",
        "∑∑": "the double summation of",
        "∏": "the product of",
        "∂": "the partial derivative of",
        "∇": "the gradient of",
        "∆": "delta",
        "Δ": "delta",

        # Greek letters (common in math/science)
        "θ": "theta",
        "π": "pi",
        "φ": "phi",
        "ψ": "psi",
        "λ": "lambda",
        "μ": "mu",
        "σ": "sigma",
        "ω": "omega",
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ε": "epsilon",
        "ζ": "zeta",
        "η": "eta",
        "κ": "kappa",
        "ν": "nu",
        "ξ": "xi",
        "ρ": "rho",
        "τ": "tau",
        "υ": "upsilon",
        "χ": "chi",

        # Root and comparison symbols
        "√": "the square root of",
        "∛": "the cube root of",
        "≈": "approximately equal to",
        "≤": "less than or equal to",
        "≥": "greater than or equal to",
        "≠": "not equal to",
        "≡": "identical to",

        # Arrows and relations
        "→": "tends to",
        "⇒": "implies",
        "⇐": "is implied by",
        "⇔": "if and only if",
        "∞": "infinity",

        # Operators
        "÷": "divided by",
        "×": "times",
        "±": "plus or minus",
        "∓": "minus or plus",

        # Logic symbols
        "∴": "therefore",
        "∵": "because",
        "∃": "there exists",
        "∀": "for all",
        "∈": "in",
        "∉": "not in",
        "⊂": "subset of",
        "⊆": "subset of or equal to",
        "∪": "union",
        "∩": "intersection",

        # Set theory
        "∅": "empty set",
        "ℝ": "the set of real numbers",
        "ℕ": "the set of natural numbers",
        "ℤ": "the set of integers",
        "ℚ": "the set of rational numbers",
        "ℂ": "the set of complex numbers",

        # Special constants and functions
        "e": "the mathematical constant e",
        "f(x)": "f of x",
        "g(x)": "g of x",
        "sin": "sine",
        "cos": "cosine",
        "tan": "tangent",
        "log": "logarithm",
        "ln": "natural logarithm",
        "lim": "limit",
    }

    # Regex patterns for complex structures
    EQUATION_PATTERNS = [
        # Subscripts: x_2 -> x sub 2
        (r"([a-zA-Z])_([0-9]+)", r"\1 sub \2"),
        # Powers: x^2 -> x to the power of 2
        (r"([a-zA-Z])\^([0-9]+)", r"\1 to the power of \2"),
        # Definite integrals: ∫_a^b f(x)dx -> the integral from a to b of f of x
        (r"∫_([a-zA-Z0-9]+)\^([a-zA-Z0-9]+)(.+?)dx", r"the integral from \1 to \2 of \3 with respect to x"),
        # Indefinite integrals: ∫ f(x)dx -> the integral of f of x with respect to x
        (r"∫\s*(.+?)dx", r"the integral of \1 with respect to x"),
        # Simple integrals without dx
        (r"∫\s*(.+)", r"the integral of \1"),
        # Fractions: a/b -> a divided by b
        (r"(\d+)/(\d+)", r"\1 divided by \2"),
        # Variable fractions: x/y -> x divided by y
        (r"([a-zA-Z]+)/([a-zA-Z])", r"\1 divided by \2"),
        # Functions with arguments: f(x) -> f of x
        (r"([a-zA-Z])\(([a-zA-Z0-9,]+)\)", r"\1 of \2"),
        # Derivatives: dy/dx -> d y by d x
        (r"d([a-zA-Z])/d([a-zA-Z])", r"the derivative of \1 with respect to \2"),
        # Second derivatives: d^2y/dx^2 -> the second derivative of y with respect to x
        (r"d\^2([a-zA-Z])/dx\^2", r"the second derivative of \1 with respect to x"),
    ]

    @classmethod
    def linearize(cls, text: Optional[str]) -> str:
        """
        Convert mathematical notation to spoken form.

        Args:
            text: Text with mathematical notation

        Returns:
            Text with math converted to speech

        Examples:
            >>> MathLinearizer.linearize("x^2 + y^2 = r^2")
            'x to the power of 2 plus y to the power of 2 equals r squared'
        """
        if text is None:
            logger.warning("MathLinearizer received None, returning empty string")
            return ""

        if not isinstance(text, str):
            logger.warning(f"MathLinearizer received non-string type {type(text)}, converting to string")
            try:
                text = str(text)
            except Exception as e:
                logger.error(f"Failed to convert input to string: {e}")
                return ""

        if not text.strip():
            return text

        try:
            result = text

            # Step 1: Replace complex patterns before simple symbols
            # This prevents partial matches from breaking complex structures
            for pattern, replacement in cls.EQUATION_PATTERNS:
                try:
                    result = re.sub(pattern, replacement, result)
                except re.error as e:
                    logger.error(f"Regex error for pattern '{pattern}': {e}")

            # Step 2: Replace simple symbols
            for symbol, spoken in cls.MATH_SYMBOLS.items():
                # If the symbol is alphabetical (e.g., 'e', 'sin', 'tan'), use regex word boundaries
                if symbol.isalpha():
                    # \b ensures we only match whole words, so 'e' doesn't trigger inside 'teacher'
                    # We use re.IGNORECASE if you want it to catch 'Sin' and 'sin', but standard is fine here.
                    pattern = rf'\b{symbol}\b'
                    result = re.sub(pattern, spoken, result)
                else:
                    # Non-alphabetical symbols (like ∫, +, =) are safe for standard replace
                    result = result.replace(symbol, spoken)

            # Step 3: Linearize equation structures
            result = cls._linearize_equations(result)

            # Step 4: Post-process to fix common issues
            result = cls._post_process(result)

            return result

        except Exception as e:
            logger.error(f"Error linearizing text: {e}", exc_info=True)
            # Return original text on error
            return text

    @classmethod
    def _linearize_equations(cls, text: str) -> str:
        """
        Linearize equation structures with error handling.

        Args:
            text: Text containing equations

        Returns:
            Linearized text
        """
        if not text:
            return text

        try:
            # f(x) → f of x (already handled by patterns, but keeping as fallback)
            text = re.sub(r"([a-zA-Z])\(([a-zA-Z])\)", r"\1 of \2", text)

            # = → equals
            text = text.replace("=", "equals")

            # + → plus, - → minus
            text = text.replace("+", "plus").replace("-", "minus")

            # * → times, / → divided by
            text = text.replace("*", "times").replace("/", "divided by")

            # Handle common phrase combinations
            text = text.replace("to the power of 2 plus", "squared plus")
            text = text.replace("to the power of 3 plus", "cubed plus")

            return text

        except Exception as e:
            logger.error(f"Error linearizing equations: {e}", exc_info=True)
            return text

    @classmethod
    def _post_process(cls, text: str) -> str:
        """
        Post-process text to fix common issues and improve readability.

        Args:
            text: Linearized text

        Returns:
            Cleaned up text
        """
        if not text:
            return text

        try:
            # Fix multiple spaces
            text = re.sub(r'\s+', ' ', text)

            # Fix "sub  " with extra spaces
            text = text.replace("sub  ", "sub ")

            # Fix "to the power of" extra spaces
            text = re.sub(r'the\s+integral\s+of', 'the integral of', text)

            # Fix "divided by" extra spaces
            text = re.sub(r'divided\s+by', 'divided by', text)

            # Ensure proper spacing around operators
            text = re.sub(r'\s+plus\s+', ' plus ', text)
            text = re.sub(r'\s+minus\s+', ' minus ', text)
            text = re.sub(r'\s+times\s+', ' times ', text)

            # Remove leading/trailing whitespace
            text = text.strip()

            return text

        except Exception as e:
            logger.error(f"Error in post-processing: {e}", exc_info=True)
            return text

    @classmethod
    def linearize_batch(cls, texts: list) -> list:
        """
        Linearize a batch of texts.

        Args:
            texts: List of texts with mathematical notation

        Returns:
            List of linearized texts
        """
        if not isinstance(texts, list):
            logger.error("linearize_batch received non-list input")
            return []

        results = []
        for i, text in enumerate(texts):
            try:
                result = cls.linearize(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error linearizing text at index {i}: {e}", exc_info=True)
                results.append(text)  # Append original on error

        return results
