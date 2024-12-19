import uuid
import random
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
import numpy as np
from dataclasses import dataclass
from fragments import ConceptualFragment, TreatyFragment

class AgentType(Enum):
    EXPLORER = "explorer"
    CONSERVATIVE = "conservative"
    NEGOTIATOR = "negotiator"
    SPECIALIST = "specialist"
    BASIC = "basic"

class AgentSpecialization(Enum):
    GENERAL = "general"
    LOGICAL = "logical_reasoner"
    SOCIAL = "social"
    PROBLEM_SOLVER = "problem_solver"

class EnvironmentType(Enum):
    LOGICAL = "logical"
    PROBLEM_SOLVING = "problem_solving"
    SOCIAL = "social"

##########################################
# Enhanced Data Structures 
##########################################

@dataclass
class InteractionHistory:
    """Track detailed interaction history between agents."""
    timestamp: float
    agent_a_id: uuid.UUID
    agent_b_id: uuid.UUID
    interaction_type: str
    success: bool
    fragment_types: List[str]
    similarity_score: float

class GameTheoryPayoff:
    """Represents payoff matrices for agent interactions."""
    def __init__(self):
        # Default payoff matrix for prisoner's dilemma-like scenarios
        self.payoff_matrix = {
            ("cooperate", "cooperate"): (3, 3),
            ("cooperate", "defect"): (0, 5),
            ("defect", "cooperate"): (5, 0),
            ("defect", "defect"): (1, 1)
        }
    
    def get_payoff(self, strategy_a: str, strategy_b: str) -> Tuple[float, float]:
        return self.payoff_matrix.get((strategy_a, strategy_b), (0, 0))

class PretrainedEmbeddings:
    """Simulated pretrained word embeddings."""
    def __init__(self, embedding_dim=100):
        self.embedding_dim = embedding_dim
        self.word_vectors = {}
        self.vocabulary = set()
        
    def load_pretrained(self, path: Optional[str] = None):
        """Simulate loading pretrained embeddings."""
        # In a real implementation, this would load actual pretrained vectors
        common_words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i"] 
        for word in common_words:
            self.word_vectors[word] = np.random.normal(0, 0.1, self.embedding_dim)
            self.vocabulary.add(word)
    
    def get_vector(self, word: str) -> np.ndarray:
        word = word.lower()
        if word not in self.word_vectors:
            # Generate a stable random vector for unknown words
            random.seed(hash(word))
            self.word_vectors[word] = np.random.normal(0, 0.1, self.embedding_dim)
            self.vocabulary.add(word)
        return self.word_vectors[word]

# Global pretrained embeddings instance
pretrained_embeddings = PretrainedEmbeddings()
pretrained_embeddings.load_pretrained()

class Strategy(Enum):
    """Game theory strategies for agent interactions."""
    COOPERATE = "cooperate"
    DEFECT = "defect"
    TIT_FOR_TAT = "tit_for_tat"
    PAVLOV = "pavlov"
    RANDOM = "random"

@dataclass
class InteractionResult:
    """Stores the result of an agent interaction."""
    success: bool
    payoff: float
    fragments_created: List[ConceptualFragment]
    strategy_used: Strategy
    energy_delta: float
    treaties_formed: List[TreatyFragment]
