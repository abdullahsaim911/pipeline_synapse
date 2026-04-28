"""
Knowledge Base Module

Stores cross-domain knowledge for connecting concepts.
Handles edge cases and provides robust error handling.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Manages cross-domain knowledge and relationships."""

    # Comprehensive cross-domain knowledge mapping
    CROSS_DOMAIN_KNOWLEDGE = {
        "derivative": {
            "math": "rate of change, slope of curve, instantaneous rate",
            "physics": "velocity from position, acceleration from velocity, momentum change",
            "chemistry": "reaction rate, concentration gradient, thermodynamic derivatives",
            "economics": "marginal cost, marginal utility, optimization, elasticity",
            "biology": "population growth rate, enzyme kinetics",
            "computer_science": "gradient descent, backpropagation, algorithm complexity"
        },
        "integral": {
            "math": "area under curve, accumulation, antiderivative, sum of infinitesimals",
            "physics": "displacement from velocity, work from force, charge from current",
            "economics": "total cost from marginal cost, consumer surplus, producer surplus",
            "chemistry": "total concentration change, thermodynamic work, entropy change",
            "biology": "total population growth, metabolic accumulation",
            "computer_science": "cumulative distribution function, expected value calculation"
        },
        "linear_function": {
            "math": "constant rate of change, straight line, y equals m x plus b, proportional",
            "physics": "uniform motion, constant velocity or acceleration, Hooke's law",
            "chemistry": "first order kinetics, constant reaction rate, Beer-Lambert law",
            "biology": "constant population growth, linear DNA replication",
            "economics": "linear cost functions, fixed and variable costs",
            "computer_science": "linear time complexity, linear search"
        },
        "slope": {
            "math": "rate of change, derivative, rise over run, steepness",
            "physics": "velocity from position, acceleration from velocity, gradient",
            "economics": "marginal rate of substitution, marginal cost",
            "biology": "growth rate gradient",
            "geography": "topographical gradient, elevation change"
        },
        "rate": {
            "math": "derivative, rate of change, ratio",
            "physics": "speed, velocity, acceleration, frequency",
            "chemistry": "reaction rate, rate law, catalysis rate",
            "economics": "growth rate, inflation rate, interest rate",
            "biology": "metabolic rate, heart rate, population growth rate"
        },
        "proportion": {
            "math": "ratio, fraction, equal ratios, scaling factor",
            "physics": "direct relationship, inverse relationship, similarity",
            "chemistry": "mole ratio, stoichiometric ratio, concentration proportion",
            "biology": "population ratio, genetic inheritance ratios",
            "art": "golden ratio, perspective, scaling"
        },
        "exponential": {
            "math": "rapid growth, exponential function, compound growth",
            "physics": "radioactive decay, capacitor discharge, population dynamics",
            "chemistry": "first-order reactions, concentration decay",
            "economics": "compound interest, economic growth models",
            "biology": "bacterial growth, virus spread, population explosion"
        },
        "logarithm": {
            "math": "inverse of exponential, log scale, orders of magnitude",
            "physics": "decibel scale, pH scale, Richter scale",
            "chemistry": "pH calculation, acid-base equilibrium",
            "computer_science": "algorithmic complexity, information theory",
            "economics": "log-linear models, elasticity measurements"
        },
        "probability": {
            "math": "likelihood, chance, random variable, distribution",
            "physics": "quantum mechanics, statistical mechanics, entropy",
            "chemistry": "reaction probability, collision theory",
            "economics": "risk analysis, expected value, decision theory",
            "biology": "genetic inheritance, mutation rates, population genetics"
        },
        "function": {
            "math": "mapping, input-output relationship, transformation",
            "physics": "relationship between variables, state function",
            "chemistry": "thermodynamic functions, state variables",
            "computer_science": "procedure, method, algorithmic mapping",
            "economics": "demand function, supply function, utility function"
        },
        "vector": {
            "math": "direction and magnitude, array, multi-dimensional quantity",
            "physics": "force, velocity, acceleration, field",
            "chemistry": "dipole moment, molecular orientation",
            "computer_science": "data structures, embeddings, feature vectors",
            "biology": "gene expression vectors, directional movement"
        },
        "matrix": {
            "math": "array of numbers, linear transformation, system of equations",
            "physics": "quantum mechanics, stress tensor, rotation matrices",
            "chemistry": "molecular symmetry operations, quantum chemistry",
            "computer_science": "neural networks, graph theory, data representation",
            "economics": "input-output models, game theory matrices"
        },
        "equilibrium": {
            "math": "stable point, balance point, solution to system",
            "physics": "static equilibrium, dynamic equilibrium, forces balance",
            "chemistry": "chemical equilibrium, reaction balance, Le Chatelier's principle",
            "economics": "market equilibrium, supply-demand balance, Nash equilibrium",
            "biology": "homeostasis, ecological balance, population equilibrium"
        },
        "optimization": {
            "math": "maximization, minimization, finding extrema, calculus of variations",
            "physics": "principle of least action, energy minimization",
            "chemistry": "transition state theory, reaction pathway optimization",
            "economics": "profit maximization, cost minimization, utility maximization",
            "computer_science": "algorithm optimization, machine learning loss functions"
        },
        "symmetry": {
            "math": "invariance under transformation, reflection, rotation",
            "physics": "conservation laws, crystal structure, particle physics",
            "chemistry": "molecular symmetry, crystallography, spectroscopy",
            "biology": "bilateral symmetry, radial symmetry, molecular biology",
            "art": "visual balance, composition, aesthetics"
        },
        "wave": {
            "math": "periodic function, oscillation, sine and cosine",
            "physics": "sound waves, electromagnetic waves, quantum wavefunctions",
            "chemistry": "spectroscopy, molecular vibrations, wave mechanics",
            "computer_science": "signal processing, Fourier transforms",
            "economics": "business cycles, market fluctuations"
        },
        "entropy": {
            "math": "measure of disorder, information content, logarithmic measure",
            "physics": "second law of thermodynamics, disorder, heat death",
            "chemistry": "spontaneity of reactions, Gibbs free energy",
            "computer_science": "information theory, data compression",
            "economics": "market efficiency, information asymmetry"
        }
    }

    @staticmethod
    def get_related_domains(concept: Optional[str]) -> List[str]:
        """
        Get all domains where this concept appears.

        Args:
            concept: Concept name (e.g., "derivative")

        Returns:
            List of domain names where concept is relevant

        Examples:
            >>> KnowledgeBase.get_related_domains("derivative")
            ['math', 'physics', 'chemistry', 'economics', 'biology', 'computer_science']
        """
        if not concept or not isinstance(concept, str):
            logger.warning(f"get_related_domains received invalid concept: {concept}")
            return []

        domains = []
        concept_lower = concept.lower().strip()

        # Check all concept entries for matches
        for concept_key, concept_data in KnowledgeBase.CROSS_DOMAIN_KNOWLEDGE.items():
            # Check if the concept matches the key or appears in any domain description
            if concept_lower == concept_key:
                domains.extend(concept_data.keys())
            else:
                # Check if concept appears in any domain's explanation
                for domain, explanation in concept_data.items():
                    if concept_lower in explanation.lower() and domain not in domains:
                        domains.append(domain)

        # Remove duplicates while preserving order
        seen = set()
        unique_domains = []
        for domain in domains:
            if domain not in seen:
                seen.add(domain)
                unique_domains.append(domain)

        return unique_domains

    @staticmethod
    def get_explanation(concept: Optional[str], from_domain: Optional[str],
                       to_domain: Optional[str]) -> str:
        """
        Get explanation of concept in another domain.

        Args:
            concept: Concept name (e.g., "derivative")
            from_domain: Source domain (for validation/context)
            to_domain: Target domain (e.g., "physics")

        Returns:
            Explanation string, empty string if not found

        Examples:
            >>> KnowledgeBase.get_explanation("derivative", "math", "physics")
            'velocity from position, acceleration from velocity, momentum change'
        """
        if not concept or not isinstance(concept, str):
            logger.warning(f"get_explanation received invalid concept: {concept}")
            return ""

        if not to_domain or not isinstance(to_domain, str):
            logger.warning(f"get_explanation received invalid to_domain: {to_domain}")
            return ""

        concept_lower = concept.lower().strip()
        to_domain_lower = to_domain.lower().strip()

        # Direct lookup
        concept_data = KnowledgeBase.CROSS_DOMAIN_KNOWLEDGE.get(concept_lower, {})
        explanation = concept_data.get(to_domain_lower, "")

        # If not found directly, try fuzzy matching
        if not explanation:
            logger.debug(f"No direct explanation found for '{concept}' in '{to_domain}', trying fuzzy match")
            for concept_key, concept_data in KnowledgeBase.CROSS_DOMAIN_KNOWLEDGE.items():
                if concept_lower in concept_key or concept_key in concept_lower:
                    explanation = concept_data.get(to_domain_lower, "")
                    if explanation:
                        logger.debug(f"Found fuzzy match using key: {concept_key}")
                        break

        return explanation if explanation else ""

    @staticmethod
    def get_all_concepts() -> List[str]:
        """
        Get list of all available concepts in the knowledge base.

        Returns:
            List of concept names
        """
        return list(KnowledgeBase.CROSS_DOMAIN_KNOWLEDGE.keys())

    @staticmethod
    def get_domains_for_concept(concept: Optional[str]) -> Dict[str, str]:
        """
        Get all domains and their explanations for a given concept.

        Args:
            concept: Concept name

        Returns:
            Dictionary mapping domains to their explanations
        """
        if not concept or not isinstance(concept, str):
            logger.warning(f"get_domains_for_concept received invalid concept: {concept}")
            return {}

        concept_lower = concept.lower().strip()
        concept_data = KnowledgeBase.CROSS_DOMAIN_KNOWLEDGE.get(concept_lower, {})

        # Try fuzzy match if direct lookup fails
        if not concept_data:
            for concept_key, concept_data in KnowledgeBase.CROSS_DOMAIN_KNOWLEDGE.items():
                if concept_lower in concept_key or concept_key in concept_lower:
                    return concept_data

        return concept_data

    @staticmethod
    def find_concepts_by_domain(domain: Optional[str]) -> List[str]:
        """
        Find all concepts that have explanations in a specific domain.

        Args:
            domain: Domain name (e.g., "physics")

        Returns:
            List of concept names relevant to the domain
        """
        if not domain or not isinstance(domain, str):
            logger.warning(f"find_concepts_by_domain received invalid domain: {domain}")
            return []

        domain_lower = domain.lower().strip()
        concepts = []

        for concept, concept_data in KnowledgeBase.CROSS_DOMAIN_KNOWLEDGE.items():
            if domain_lower in concept_data:
                concepts.append(concept)

        return concepts

    @staticmethod
    def add_custom_knowledge(concept: str, domain: str, explanation: str) -> bool:
        """
        Add or update custom knowledge entry.

        Args:
            concept: Concept name
            domain: Domain name
            explanation: Explanation text

        Returns:
            True if successful, False otherwise
        """
        if not all([concept, domain, explanation]) or not all(isinstance(x, str) for x in [concept, domain, explanation]):
            logger.error("add_custom_knowledge received invalid parameters")
            return False

        try:
            concept_lower = concept.lower().strip()
            domain_lower = domain.lower().strip()

            if concept_lower not in KnowledgeBase.CROSS_DOMAIN_KNOWLEDGE:
                KnowledgeBase.CROSS_DOMAIN_KNOWLEDGE[concept_lower] = {}

            KnowledgeBase.CROSS_DOMAIN_KNOWLEDGE[concept_lower][domain_lower] = explanation
            logger.info(f"Added knowledge: {concept} -> {domain}")
            return True

        except Exception as e:
            logger.error(f"Error adding custom knowledge: {e}", exc_info=True)
            return False
