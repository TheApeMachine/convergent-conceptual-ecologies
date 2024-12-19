from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import uuid
from step import EnvironmentType, AgentSpecialization
from conceptual_fragment import ConceptualFragment, TextFragment, VectorFragment, TreatyFragment
from ecologies import ConceptualEcology
import numpy as np
from datetime import datetime
from collections import defaultdict

class VirtualEnvironment(ABC):
    def __init__(self, name: Optional[str] = None, 
                 env_type: EnvironmentType = EnvironmentType.LOGICAL,
                 specialization: AgentSpecialization = AgentSpecialization.GENERAL):
        self.id = uuid.uuid4()
        self.name = name if name else f"Env-{self.id}"
        self.env_type = env_type
        self.specialization = specialization

    @abstractmethod
    def evaluate(self, fragments: List[ConceptualFragment], ecology: 'ConceptualEcology') -> Dict:
        pass


class EnhancedEnvironment(VirtualEnvironment):
    """Base class for enhanced environments with advanced evaluation."""
    
    def __init__(self, name: Optional[str] = None,
                 env_type: EnvironmentType = EnvironmentType.LOGICAL,
                 specialization: AgentSpecialization = AgentSpecialization.GENERAL):
        super().__init__(name, env_type, specialization)
        self.evaluation_history: List[Dict] = []
        self.adaptation_rate = 0.1
        self.complexity = 1.0

    def evaluate(self, fragments: List[ConceptualFragment], ecology: 'ConceptualEcology') -> Dict:
        """Base evaluation method - should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement evaluate()")

    def record_evaluation(self, fragments: List[ConceptualFragment],
                         score: float, metadata: Dict):
        """Record evaluation results for analysis."""
        self.evaluation_history.append({
            'timestamp': datetime.now(),
            'fragment_count': len(fragments),
            'fragment_types': [f.__class__.__name__ for f in fragments],
            'score': score,
            'metadata': metadata,
            'complexity': self.complexity
        })

        # Trim history if it gets too long
        if len(self.evaluation_history) > 1000:
            self.evaluation_history = self.evaluation_history[-1000:]

    def adapt_difficulty(self):
        """Adjust environment complexity based on evaluation history."""
        if len(self.evaluation_history) < 5:
            return
        
        # Look at recent performance
        recent_scores = [entry['score'] for entry in self.evaluation_history[-5:]]
        avg_score = sum(recent_scores) / len(recent_scores)
        
        # Adjust complexity based on performance
        if avg_score > 0.8:  # Too easy
            self.complexity *= (1 + self.adaptation_rate)
        elif avg_score < 0.2:  # Too hard
            self.complexity *= (1 - self.adaptation_rate)
        
        # Keep complexity within reasonable bounds
        self.complexity = max(0.5, min(5.0, self.complexity))

    def get_performance_stats(self) -> Dict:
        """Get statistical summary of environment performance."""
        if not self.evaluation_history:
            return {}
        
        recent_entries = self.evaluation_history[-100:]  # Last 100 evaluations
        scores = [entry['score'] for entry in recent_entries]
        
        return {
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'current_complexity': self.complexity,
            'num_evaluations': len(self.evaluation_history),
            'fragment_type_distribution': self._get_fragment_distribution(recent_entries)
        }

    def _get_fragment_distribution(self, entries: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of fragment types in recent evaluations."""
        distribution = defaultdict(int)
        for entry in entries:
            for fragment_type in entry['fragment_types']:
                distribution[fragment_type] += 1
        return dict(distribution)

    def reset_complexity(self):
        """Reset environment complexity to default."""
        self.complexity = 1.0
        
    def __repr__(self):
        return (f"<{self.__class__.__name__} {self.name} | "
                f"type={self.env_type.value} spec={self.specialization.value} | "
                f"complexity={self.complexity:.2f}>")

class LogicalEnvironment(EnhancedEnvironment):
    """Enhanced logical environment with complexity adaptation."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(
            name=name,
            env_type=EnvironmentType.LOGICAL,
            specialization=AgentSpecialization.LOGICAL
        )
        self.consistency_weight = 0.6
        self.treaty_weight = 0.4

    def evaluate(self, fragments: List[ConceptualFragment], ecology: 'ConceptualEcology') -> Dict:
        if not fragments:
            return {"consistency_score": 0.0}
        
        # Evaluate logical consistency
        text_fragments = [f for f in fragments if isinstance(f, TextFragment)]
        if not text_fragments:
            return {"consistency_score": 0.0}
        
        # Analyze text structure and patterns
        scores = []
        for f in text_fragments:
            # Assess complexity and structure
            complexity_score = f.complexity_score
            keyword_score = len(f.keywords) / 10  # Normalize by expected max keywords
            
            # Consider treaty stability
            treaty_bonus = 0.0
            if ecology.treaty_store.is_stable(f):
                treaty_bonus = 0.3
            
            # Combined score with complexity adjustment
            fragment_score = (
                self.consistency_weight * (complexity_score + keyword_score) +
                self.treaty_weight * treaty_bonus
            ) / self.complexity
            
            scores.append(fragment_score)
        
        final_score = sum(scores) / len(scores)
        
        # Record evaluation
        self.record_evaluation(
            fragments=fragments,
            score=final_score,
            metadata={
                'complexity': self.complexity,
                'fragment_count': len(fragments),
                'text_fragment_count': len(text_fragments)
            }
        )
        
        # Adapt difficulty
        self.adapt_difficulty()
        
        return {"consistency_score": min(1.0, final_score)}

class ProblemSolvingEnvironment(EnhancedEnvironment):
    """Enhanced problem-solving environment with dynamic challenges."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(
            name=name,
            env_type=EnvironmentType.PROBLEM_SOLVING,
            specialization=AgentSpecialization.PROBLEM_SOLVER
        )
        self.target_vector = np.random.normal(0, 1, 5)  # Initial problem vector
        self.problem_history: List[Dict] = []
        self.solution_threshold = 0.7

    def evaluate(self, fragments: List[ConceptualFragment], ecology: 'ConceptualEcology') -> Dict:
        if not fragments:
            return {"consistency_score": 0.0}
        
        # Focus on vector fragments for problem-solving
        vector_fragments = [f for f in fragments if isinstance(f, VectorFragment)]
        if not vector_fragments:
            return {"consistency_score": 0.0}
        
        # Evaluate solutions
        scores = []
        for fragment in vector_fragments:
            # Calculate solution quality
            solution_vector = np.array(fragment.data)
            if len(solution_vector) != len(self.target_vector):
                continue
                
            # Multiple evaluation metrics
            cosine_sim = np.dot(solution_vector, self.target_vector) / (
                np.linalg.norm(solution_vector) * np.linalg.norm(self.target_vector)
            )
            
            euclidean_dist = np.linalg.norm(solution_vector - self.target_vector)
            euclidean_sim = 1.0 / (1.0 + euclidean_dist)
            
            # Combined score adjusted for complexity
            score = (0.7 * cosine_sim + 0.3 * euclidean_sim) / self.complexity
            scores.append(score)
        
        if not scores:
            return {"consistency_score": 0.0}
        
        # Take best solution
        best_score = max(scores)
        
        # Record problem-solving attempt
        self.problem_history.append({
            'timestamp': datetime.now(),
            'target_vector': self.target_vector.tolist(),
            'best_score': best_score,
            'num_attempts': len(vector_fragments)
        })
        
        # Adapt problem difficulty
        if best_score > self.solution_threshold:
            # Generate new problem
            self.target_vector = np.random.normal(0, self.complexity, 5)
        
        # Record evaluation
        self.record_evaluation(
            fragments=fragments,
            score=best_score,
            metadata={
                'complexity': self.complexity,
                'solution_attempts': len(vector_fragments)
            }
        )
        
        # Adapt difficulty
        self.adapt_difficulty()
        
        return {"consistency_score": min(1.0, best_score)}

class SocialEnvironment(EnhancedEnvironment):
    """Enhanced social environment rewarding cooperation and treaty formation."""
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(
            name=name,
            env_type=EnvironmentType.SOCIAL,
            specialization=AgentSpecialization.SOCIAL
        )
        self.treaty_weight = 0.6
        self.cooperation_weight = 0.4
        self.min_agreement_threshold = 0.5

    def evaluate(self, fragments: List[ConceptualFragment], ecology: 'ConceptualEcology') -> Dict:
        if not fragments or not ecology.agents:
            return {"consistency_score": 0.0}
        
        # Evaluate social impact and treaty formation
        treaty_score = 0.0
        cooperation_score = 0.0
        
        # Analyze treaty participation and stability
        treaties = [f for f in fragments if isinstance(f, TreatyFragment)]
        if treaties:
            treaty_validations = [t.validate(ecology.agents) for t in treaties]
            treaty_score = sum(1.0 for v in treaty_validations if v) / len(treaties)
        
        # Analyze cooperation through fragment similarity
        for fragment in fragments:
            # Check how many other agents have similar fragments
            similar_count = 0
            total_count = 0
            
            for agent in ecology.agents:
                if agent.fragments:  # Skip agents without fragments
                    max_similarity = max(
                        fragment.similarity(f) for f in agent.fragments
                    )
                    if max_similarity > self.min_agreement_threshold:
                        similar_count += 1
                    total_count += 1
            
            if total_count > 0:
                cooperation_score += similar_count / total_count
        
        cooperation_score = cooperation_score / len(fragments) if fragments else 0
        
        # Combined social score
        final_score = (
            self.treaty_weight * treaty_score +
            self.cooperation_weight * cooperation_score
        ) / self.complexity
        
        # Record evaluation
        self.record_evaluation(
            fragments=fragments,
            score=final_score,
            metadata={
                'complexity': self.complexity,
                'treaty_score': treaty_score,
                'cooperation_score': cooperation_score,
                'fragment_count': len(fragments),
                'treaty_count': len(treaties)
            }
        )
        
        # Adapt difficulty
        self.adapt_difficulty()
        
        return {"consistency_score": min(1.0, final_score)}
