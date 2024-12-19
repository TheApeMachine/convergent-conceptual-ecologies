from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
import uuid
from datetime import datetime
import numpy as np
import math
from agents import ConceptualAgent

class ConceptualFragment(ABC):
    """Enhanced abstract class for conceptual fragments."""
    def __init__(self, data: Any):
        self.id = uuid.uuid4()
        self.data = data
        self.quality = 0.0
        self.usage_count = 0
        self.durability = 0
        self.creation_generation = 0
        self.last_used = datetime.now()
        self.creator_id: Optional[uuid.UUID] = None
        self.modification_history: List[Dict] = []
        self.tags: Set[str] = set()

    @abstractmethod
    def similarity(self, other: 'ConceptualFragment') -> float:
        pass

    @abstractmethod
    def merge(self, other: 'ConceptualFragment') -> 'ConceptualFragment':
        pass

    def update_quality(self, score: float):
        self.quality = max(0.0, min(1.0, self.quality + score))
        self.modification_history.append({
            'timestamp': datetime.now(),
            'type': 'quality_update',
            'old_value': self.quality - score,
            'new_value': self.quality
        })

    def on_usage(self):
        self.usage_count += 1
        self.durability += 1
        self.last_used = datetime.now()

    def add_tag(self, tag: str):
        self.tags.add(tag)

    def to_dict(self) -> Dict:
        """Serialize fragment to dictionary for storage/transmission."""
        return {
            'id': str(self.id),
            'type': self.__class__.__name__,
            'data': self.data,
            'quality': self.quality,
            'usage_count': self.usage_count,
            'durability': self.durability,
            'creation_generation': self.creation_generation,
            'last_used': self.last_used.isoformat(),
            'creator_id': str(self.creator_id) if self.creator_id else None,
            'tags': list(self.tags)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ConceptualFragment':
        """Deserialize fragment from dictionary."""
        fragment = cls(data['data'])
        fragment.id = uuid.UUID(data['id'])
        fragment.quality = data['quality']
        fragment.usage_count = data['usage_count']
        fragment.durability = data['durability']
        fragment.creation_generation = data['creation_generation']
        fragment.last_used = datetime.fromisoformat(data['last_used'])
        fragment.creator_id = uuid.UUID(data['creator_id']) if data['creator_id'] else None
        fragment.tags = set(data['tags'])
        return fragment

class TreatyFragment(ConceptualFragment):
    """Enhanced treaty fragment with validation and consensus tracking."""
    def __init__(self, data: str, parties: Optional[Set[uuid.UUID]] = None):
        super().__init__(data)
        self.parties = parties or set()
        self.signatures: Dict[uuid.UUID, datetime] = {}
        self.terms: List[str] = []
        self.validation_threshold = 0.7
        self.consensus_score = 0.0
        self.stability_score = 0.0
        self.last_validated = datetime.now()
        self.validation_history: List[Dict] = []

    def add_party(self, agent_id: uuid.UUID) -> None:
        """Add a party to the treaty."""
        self.parties.add(agent_id)
        self.signatures[agent_id] = datetime.now()

    def remove_party(self, agent_id: uuid.UUID) -> None:
        """Remove a party from the treaty."""
        self.parties.discard(agent_id)
        self.signatures.pop(agent_id, None)

    def add_terms(self, terms: List[str]) -> None:
        """Add terms to the treaty."""
        self.terms.extend(terms)
        self.modification_history.append({
            'timestamp': datetime.now(),
            'type': 'terms_added',
            'terms': terms
        })

    def validate(self, agents: List['ConceptualAgent']) -> bool:
        """Validate treaty against current agent population."""
        if not self.parties or not agents:
            return False

        # Calculate agreement ratio
        agreeing_agents = sum(1 for agent in agents 
                            if any(self.similarity(f) > 0.9 for f in agent.fragments))
        agreement_ratio = agreeing_agents / len(agents)
        
        # Update stability metrics
        self.consensus_score = agreement_ratio
        self.stability_score = self.consensus_score * (1.0 - math.exp(-len(self.signatures) / 5.0))
        
        # Record validation
        self.last_validated = datetime.now()
        self.validation_history.append({
            'timestamp': self.last_validated,
            'consensus_score': self.consensus_score,
            'stability_score': self.stability_score
        })
        
        return self.consensus_score >= self.validation_threshold

    def similarity(self, other: ConceptualFragment) -> float:
        """Calculate similarity with other fragments."""
        if isinstance(other, TreatyFragment):
            # For treaties, consider term overlap and party overlap
            term_overlap = len(set(self.terms) & set(other.terms)) / max(1, len(set(self.terms) | set(other.terms)))
            party_overlap = len(self.parties & other.parties) / max(1, len(self.parties | other.parties))
            return 0.7 * term_overlap + 0.3 * party_overlap
        elif isinstance(other, TextFragment):
            # Compare treaty terms with text content
            text_fragment = TextFragment(" ".join(self.terms))
            return text_fragment.similarity(other)
        return 0.0

    def merge(self, other: ConceptualFragment) -> 'ConceptualFragment':
        """Merge treaties or create new treaty from combination."""
        if isinstance(other, TreatyFragment):
            # Convert lists to sets before combining
            merged_terms = list(set(self.terms).union(set(other.terms)))
            merged_parties = self.parties.union(other.parties)
            merged = TreatyFragment(f"{self.data}&{other.data}", merged_parties)
            merged.terms = merged_terms
            merged.signatures = {**self.signatures, **other.signatures}
            merged.consensus_score = (self.consensus_score + other.consensus_score) / 2
            merged.stability_score = min(self.stability_score, other.stability_score)
            return merged
        elif isinstance(other, TextFragment):
            new_treaty = TreatyFragment(f"{self.data}&{other.data}", self.parties.copy())
            new_treaty.terms = self.terms + [other.data]
            new_treaty.signatures = self.signatures.copy()
            return new_treaty
        raise TypeError("Can only merge Treaties with Treaties or TextFragments")

class TextFragment(ConceptualFragment):
    """Enhanced text fragment with improved embeddings."""
    def __init__(self, data: str, embedding_dim: int = 100):
        super().__init__(data)
        self.embedding_dim = embedding_dim
        self.embedding = self._compute_embedding(data)
        self.sentiment_score = self._compute_sentiment()
        self.complexity_score = self._compute_complexity()
        self.keywords = self._extract_keywords()

    def _compute_embedding(self, text: str) -> np.ndarray:
        # Simple placeholder embedding until proper embedding model is integrated
        return np.random.rand(self.embedding_dim)  # Returns random vector of specified dimension

    def _compute_sentiment(self) -> float:
        """Simple sentiment analysis simulation."""
        positive_words = {'good', 'great', 'excellent', 'positive', 'nice'}
        negative_words = {'bad', 'poor', 'negative', 'terrible', 'wrong'}
        
        words = self.data.lower().split()
        score = 0.0
        for word in words:
            if word in positive_words:
                score += 0.1
            elif word in negative_words:
                score -= 0.1
        return max(-1.0, min(1.0, score))

    def _compute_complexity(self) -> float:
        """Compute text complexity score based on word length and sentence structure."""
        if not self.data:
            return 0.0
        
        words = self.data.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        sentences = self.data.split('.')
        avg_sentence_length = sum(len(sent.split()) for sent in sentences) / len(sentences) if sentences else 0
        
        # Normalize scores between 0 and 1
        word_complexity = min(1.0, avg_word_length / 10.0)  # Assume max avg word length of 10
        sentence_complexity = min(1.0, avg_sentence_length / 20.0)  # Assume max avg sentence length of 20
        
        return (word_complexity + sentence_complexity) / 2.0

    def merge(self, other: ConceptualFragment) -> 'ConceptualFragment':
        """Merge two text fragments."""
        if not isinstance(other, TextFragment):
            raise TypeError("Can only merge with another TextFragment")
        merged_text = f"{self.data} {other.data}"
        merged = TextFragment(merged_text, self.embedding_dim)
        merged.tags = self.tags.union(other.tags)
        merged.quality = (self.quality + other.quality) / 2
        return merged

class VectorFragment(ConceptualFragment):
    """Enhanced vector-based conceptual fragment."""
    def __init__(self, data: List[float], dimension_labels: Optional[List[str]] = None):
        super().__init__(data)
        if not isinstance(data, list):
            raise TypeError("Vector data must be a list")
        self.dimensionality = len(data)
        self.dimension_labels = dimension_labels or [f"dim_{i}" for i in range(self.dimensionality)]
        self.vector = np.array(data)
        self.magnitude = np.linalg.norm(self.vector)
        self.normalized_vector = self.vector / self.magnitude if self.magnitude > 0 else self.vector

    def similarity(self, other: ConceptualFragment) -> float:
        if not isinstance(other, VectorFragment) or self.dimensionality != other.dimensionality:
            return 0.0
        
        # Enhanced similarity metrics
        cosine_sim = np.dot(self.normalized_vector, other.normalized_vector)
        
        # Euclidean similarity (inversely proportional to distance)
        euclidean_dist = np.linalg.norm(self.vector - other.vector)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        
        # Magnitude similarity
        mag_sim = 1.0 - abs(self.magnitude - other.magnitude) / max(self.magnitude, other.magnitude)
        
        # Weighted combination
        return 0.5 * cosine_sim + 0.3 * euclidean_sim + 0.2 * mag_sim

    def merge(self, other: ConceptualFragment) -> ConceptualFragment:
        if not isinstance(other, VectorFragment) or self.dimensionality != other.dimensionality:
            raise TypeError("Can only merge with VectorFragment of same dimensionality")
        
        # Intelligent merging strategy
        quality_weight_self = self.quality / (self.quality + other.quality) if (self.quality + other.quality) > 0 else 0.5
        quality_weight_other = 1.0 - quality_weight_self
        
        merged_vector = quality_weight_self * self.vector + quality_weight_other * other.vector
        merged = VectorFragment(merged_vector.tolist(), self.dimension_labels)
        
        # Combine metadata
        merged.tags = self.tags.union(other.tags)
        merged.quality = (self.quality + other.quality) / 2
        
        return merged

    def normalize(self) -> None:
        """Normalize the vector in place."""
        if self.magnitude > 0:
            self.vector = self.normalized_vector
            self.magnitude = 1.0

    def project(self, dimensions: List[int]) -> 'VectorFragment':
        """Project vector onto specified dimensions."""
        if not all(0 <= d < self.dimensionality for d in dimensions):
            raise ValueError("Invalid dimension indices")
        projected_data = [self.data[i] for i in dimensions]
        projected_labels = [self.dimension_labels[i] for i in dimensions]
        return VectorFragment(projected_data, projected_labels)
