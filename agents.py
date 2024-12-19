from abc import ABC, abstractmethod
from ast import Tuple
from collections.abc import Set
from typing import List, Dict, Any, Optional
import uuid
import random
import copy
import math
from collections import defaultdict
from environments import EnhancedEnvironment, LogicalEnvironment, ProblemSolvingEnvironment, SocialEnvironment
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import time
from conceptual_fragment import ConceptualFragment, TextFragment, VectorFragment, TreatyFragment
from scratchpad import EnvironmentType
from step import AgentType, AgentSpecialization, InteractionResult, InteractionHistory, Strategy
from abc import ABC, abstractmethod
from ecologies import ConceptualEcology

class GameTheoryPayoff:
    def get_payoff(self, strategy_a: str, strategy_b: str) -> Tuple[float, float]:
        payoff_matrix = {
            'COOPERATE': {'COOPERATE': (3, 3), 'DEFECT': (0, 5), 'TIT_FOR_TAT': (3, 3), 'RANDOM': (2, 2)},
            'DEFECT': {'COOPERATE': (5, 0), 'DEFECT': (1, 1), 'TIT_FOR_TAT': (1, 1), 'RANDOM': (2, 1)},
            'TIT_FOR_TAT': {'COOPERATE': (3, 3), 'DEFECT': (1, 1), 'TIT_FOR_TAT': (3, 3), 'RANDOM': (2, 2)},
            'RANDOM': {'COOPERATE': (2, 2), 'DEFECT': (1, 2), 'TIT_FOR_TAT': (2, 2), 'RANDOM': (1.5, 1.5)}
        }
        return payoff_matrix[strategy_a][strategy_b]

class ConceptualAgent(ABC):
    """Enhanced abstract base class for conceptual agents."""
    def __init__(self, 
                 name: Optional[str] = None, 
                 specialization: AgentSpecialization = AgentSpecialization.GENERAL,
                 agent_type: AgentType = AgentType.EXPLORER,
                 learning_rate: float = 0.1):
        self.id = uuid.uuid4()
        self.name = name if name else f"Agent-{self.id}"
        self.specialization = specialization
        self.agent_type = agent_type
        self.fragments: List[ConceptualFragment] = []
        self.energy = 100.0
        self.health = 100.0
        self.fitness = 1.0
        self.learning_rate = learning_rate
        
        # Enhanced tracking and memory
        self.interaction_history: List[InteractionHistory] = []
        self.strategy_memory: Dict[uuid.UUID, Strategy] = {}
        self.collaboration_scores: Dict[uuid.UUID, float] = defaultdict(float)
        self.lineage = [self.id]
        self.generation = 0
        self.mutation_rate = 0.1
        self.knowledge_base: Dict[str, Any] = {}
        
        # Performance metrics
        self.successful_interactions = 0
        self.failed_interactions = 0
        self.treaties_formed = 0
        self.fragments_created = 0
        
        # Learning parameters
        self.strategy_preferences = {
            strategy: 1.0 for strategy in Strategy
        }

    @abstractmethod
    def propose_fragments(self, ecology: 'ConceptualEcology') -> List[ConceptualFragment]:
        pass

    @abstractmethod
    def refine_fragments(self, environment_feedback: Dict, ecology: 'ConceptualEcology'):
        pass

    def choose_strategy(self, partner_id: uuid.UUID) -> Strategy:
        """Choose interaction strategy based on history and learning."""
        if partner_id not in self.strategy_memory:
            # No history - use weighted random choice based on preferences
            weights = list(self.strategy_preferences.values())
            strategies = list(self.strategy_preferences.keys())
            return random.choices(strategies, weights=weights)[0]
        
        # Use history to inform choice
        previous_success = self.collaboration_scores[partner_id]
        if previous_success > 0.7:
            return Strategy.COOPERATE
        elif previous_success < 0.3:
            return Strategy.DEFECT
        else:
            return Strategy.TIT_FOR_TAT

    def update_strategy_preferences(self, strategy: Strategy, success: bool):
        """Update strategy preferences based on outcomes."""
        adjustment = self.learning_rate * (1 if success else -1)
        self.strategy_preferences[strategy] += adjustment
        # Normalize preferences
        total = sum(self.strategy_preferences.values())
        self.strategy_preferences = {
            k: v/total for k, v in self.strategy_preferences.items()
        }

    def interact(self, 
                partner: 'ConceptualAgent',
                ecology: 'ConceptualEcology') -> InteractionResult:
        """Enhanced interaction logic with game theory elements."""
        if not self.fragments or not partner.fragments:
            return InteractionResult(
                success=False,
                payoff=0.0,
                fragments_created=[],
                strategy_used=Strategy.RANDOM,
                energy_delta=0.0,
                treaties_formed=[]
            )

        # Choose strategies
        my_strategy = self.choose_strategy(partner.id)
        partner_strategy = partner.choose_strategy(self.id)

        # Get payoff from game theory matrix
        payoff_matrix = GameTheoryPayoff()
        my_payoff, partner_payoff = payoff_matrix.get_payoff(
            my_strategy.value, 
            partner_strategy.value
        )

        # Attempt fragment combination based on payoff
        success = False
        new_fragments = []
        new_treaties = []
        energy_delta = my_payoff
        similarity = 0.0

        if my_payoff > 1:  # Threshold for successful interaction
            try:
                # Select compatible fragments
                my_fragment = max(self.fragments, key=lambda f: f.quality)
                partner_fragment = max(partner.fragments, key=lambda f: f.quality)
                
                # Calculate similarity
                similarity = my_fragment.similarity(partner_fragment)
                
                if similarity > 0.6:
                    merged = my_fragment.merge(partner_fragment)
                    new_fragments.append(merged)
                    
                    # Potentially form treaty
                    if similarity > 0.9:
                        treaty = TreatyFragment(
                            f"treaty_{self.id}_{partner.id}",
                            {self.id, partner.id}
                        )
                        treaty.add_terms([my_fragment.data, partner_fragment.data])
                        new_treaties.append(treaty)
                        ecology.record_treaty(treaty)
                    
                    success = True
                    energy_delta += 5.0
            except TypeError:
                success = False
                energy_delta -= 2.0

        # Update memories and scores
        self.interaction_history.append(InteractionHistory(
            timestamp=time.time(),
            agent_a_id=self.id,
            agent_b_id=partner.id,
            interaction_type=my_strategy.value,
            success=success,
            fragment_types=[f.__class__.__name__ for f in new_fragments],
            similarity_score=similarity
        ))

        # Update collaboration score
        self.collaboration_scores[partner.id] = (
            0.9 * self.collaboration_scores[partner.id] +
            0.1 * float(success)
        )

        # Update strategy preferences based on outcome
        self.update_strategy_preferences(my_strategy, success)
        
        return InteractionResult(
            success=success,
            payoff=my_payoff,
            fragments_created=new_fragments,
            strategy_used=my_strategy,
            energy_delta=energy_delta,
            treaties_formed=new_treaties
        )

    def mutate(self):
        """Enhanced mutation with multiple strategies."""
        if random.random() < self.mutation_rate:
            if self.fragments:
                # Choose random fragment to mutate
                idx = random.randint(0, len(self.fragments)-1)
                old_fragment = self.fragments[idx]
                
                # Different mutation strategies based on fragment type
                if isinstance(old_fragment, TextFragment):
                    # Text mutations: add/remove/modify words
                    mutation_type = random.choice(['add', 'remove', 'modify'])
                    if mutation_type == 'add':
                        mutated_data = f"{old_fragment.data}_new"
                    elif mutation_type == 'remove':
                        words = old_fragment.data.split('_')
                        if len(words) > 1:
                            words.pop(random.randint(0, len(words)-1))
                            mutated_data = '_'.join(words)
                        else:
                            mutated_data = old_fragment.data
                    else:  # modify
                        mutated_data = f"{old_fragment.data}_mutated"
                    
                    self.fragments[idx] = TextFragment(mutated_data)
                
                elif isinstance(old_fragment, VectorFragment):
                    # Vector mutations: gaussian noise or dimension-specific changes
                    mutation_type = random.choice(['noise', 'dimension'])
                    if mutation_type == 'noise':
                        # Add Gaussian noise
                        noise = np.random.normal(0, 0.1, len(old_fragment.data))
                        mutated_data = (np.array(old_fragment.data) + noise).tolist()
                    else:
                        # Modify specific dimension
                        mutated_data = old_fragment.data.copy()
                        dim = random.randint(0, len(mutated_data)-1)
                        mutated_data[dim] *= random.uniform(0.8, 1.2)
                    
                    self.fragments[idx] = VectorFragment(mutated_data)
                
                # Update mutation rate based on success
                if old_fragment.quality < self.fragments[idx].quality:
                    self.mutation_rate *= 0.95  # Reduce mutation rate
                else:
                    self.mutation_rate *= 1.05  # Increase mutation rate
                
                self.mutation_rate = max(0.01, min(0.5, self.mutation_rate))
                
            self.energy -= 5.0

    def reproduce(self) -> Optional['ConceptualAgent']:
        """Enhanced reproduction with trait inheritance."""
        if self.energy > 120:
            # Create child with inherited traits
            child = copy.deepcopy(self)
            child.id = uuid.uuid4()
            child.name = f"Agent-{child.id}"
            child.energy = 100.0
            child.lineage = self.lineage + [child.id]
            child.generation = self.generation + 1
            
            # Inherit successful strategies with some variation
            child.strategy_preferences = {
                strategy: pref * random.uniform(0.9, 1.1)
                for strategy, pref in self.strategy_preferences.items()
            }
            
            # Normalize strategy preferences
            total = sum(child.strategy_preferences.values())
            child.strategy_preferences = {
                k: v/total for k, v in child.strategy_preferences.items()
            }
            
            # Inherit mutation rate with possible variation
            child.mutation_rate = self.mutation_rate * random.uniform(0.9, 1.1)
            child.mutation_rate = max(0.01, min(0.5, child.mutation_rate))
            
            # Inherit some knowledge with possible mutations
            child.knowledge_base = {
                k: v * random.uniform(0.95, 1.05)
                for k, v in self.knowledge_base.items()
            }
            
            # Mutate child
            child.mutate()
            self.energy -= 30
            
            return child
        return None

    def get_fitness(self, env_feedback: Dict) -> float:
        """Enhanced fitness calculation."""
        # Base fitness from environment feedback
        consistency_score = env_feedback.get("consistency_score", 0)
        
        # Calculate fragment quality component
        fragment_quality = (
            sum(f.quality for f in self.fragments) / 
            len(self.fragments) if self.fragments else 0
        )
        
        # Calculate interaction success component
        total_interactions = (
            self.successful_interactions + self.failed_interactions
        )
        interaction_score = (
            self.successful_interactions / total_interactions 
            if total_interactions > 0 else 0
        )
        
        # Calculate treaty formation component
        treaty_score = self.treaties_formed / max(1, len(self.fragments))
        
        # Weighted combination of components
        fitness = (
            0.3 * self.health * 
            0.2 * self.energy * 
            0.2 * consistency_score * 
            0.1 * fragment_quality * 
            0.1 * interaction_score * 
            0.1 * treaty_score
        )
        
        return fitness

    def _score_fragments(self) -> List[float]:
        """Enhanced fragment scoring with multiple criteria."""
        scores = []
        for f in self.fragments:
            base_score = 0.0
            
            # Type-specific scoring
            if isinstance(f, VectorFragment):
                base_score = sum(x*x for x in f.data)
            elif isinstance(f, TextFragment):
                base_score = f.quality
            elif isinstance(f, TreatyFragment):
                base_score = f.quality + 10
            
            # Common criteria
            usage_bonus = math.log(f.usage_count + 1) * 0.1
            durability_penalty = f.durability * 0.05
            age_factor = 1.0 / (1.0 + (self.generation - f.creation_generation))
            
            # Combine scores
            final_score = (
                base_score * 0.6 + 
                usage_bonus * 0.2 +
                age_factor * 0.2 - 
                durability_penalty
            )
            
            scores.append(max(0.0, final_score))
            
        return scores

class ConservativeAgent(ConceptualAgent):
    """Enhanced conservative agent with sophisticated refinement strategies."""
    
    def __init__(self, name: Optional[str] = None,
                 specialization: AgentSpecialization = AgentSpecialization.LOGICAL):
        super().__init__(name=name, specialization=specialization,
                        agent_type=AgentType.CONSERVATIVE)
        self.refinement_threshold = 0.7
        self.quality_history: List[float] = []
        self.verified_patterns: Set[str] = set()
        self.failed_words: Set[str] = set()

    def propose_fragments(self, ecology: 'ConceptualEcology') -> List[ConceptualFragment]:
        """Propose fragments focusing on proven patterns."""
        if not self.fragments:
            return [TextFragment("conservative_initial")]
        
        # Score existing fragments
        scores = self._score_fragments()
        best_idx = scores.index(max(scores))
        best_fragment = self.fragments[best_idx]
        
        if best_fragment.quality > self.refinement_threshold:
            if isinstance(best_fragment, TextFragment):
                # Systematic refinement of text
                words = best_fragment.data.split("_")
                refined_words = [word for word in words 
                               if word not in self.failed_words]
                new_data = "_".join(refined_words) + "_refined"
                return [TextFragment(new_data)]
            
            elif isinstance(best_fragment, VectorFragment):
                # Careful vector refinement
                new_data = []
                for x in best_fragment.data:
                    if random.random() < 0.2:  # 20% chance of refinement
                        new_data.append(x + random.uniform(-0.05, 0.05))
                    else:
                        new_data.append(x)
                return [VectorFragment(new_data, best_fragment.dimension_labels)]
            
            elif isinstance(best_fragment, TreatyFragment):
                # Promote stable treaties
                if ecology.treaty_store.is_stable(best_fragment):
                    return [best_fragment]
        
        return []

    def refine_fragments(self, environment_feedback: Dict, ecology: 'ConceptualEcology'):
        """Careful refinement based on feedback with pattern learning."""
        consistency = environment_feedback.get("consistency_score", 0)
        self.quality_history.append(consistency)
        
        if len(self.quality_history) > 10:
            self.quality_history = self.quality_history[-10:]
        
        # Analyze quality trend
        trend_improving = (len(self.quality_history) > 1 and 
                         self.quality_history[-1] > self.quality_history[-2])
        
        if consistency < 0.4 or not trend_improving:
            if self.fragments:
                scores = self._score_fragments()
                worst_idx = scores.index(min(scores))
                removed_fragment = self.fragments.pop(worst_idx)
                
                # Learn from failure
                if isinstance(removed_fragment, TextFragment):
                    self.failed_words.update(removed_fragment.data.split("_"))
            self.energy -= 8
        else:
            self.energy += 12
            
        # Update refinement threshold based on success
        if trend_improving:
            self.refinement_threshold = max(0.5, self.refinement_threshold - 0.02)
        else:
            self.refinement_threshold = min(0.9, self.refinement_threshold + 0.02)
        
        # Quality updates based on treaty stability
        for fragment in self.fragments:
            if ecology.treaty_store.is_stable(fragment):
                fragment.update_quality(4)
                if isinstance(fragment, TextFragment):
                    self.verified_patterns.add(fragment.data)

class NegotiatorAgent(ConceptualAgent):
    """Enhanced negotiator agent with advanced treaty formation strategies."""
    
    def __init__(self, name: Optional[str] = None,
                 specialization: AgentSpecialization = AgentSpecialization.SOCIAL):
        super().__init__(name=name, specialization=specialization,
                        agent_type=AgentType.NEGOTIATOR)
        self.negotiation_threshold = 0.6
        self.successful_treaties: Dict[str, float] = defaultdict(float)
        self.failed_negotiations: Set[Tuple[uuid.UUID, uuid.UUID]] = set()

    def propose_fragments(self, ecology: 'ConceptualEcology') -> List[ConceptualFragment]:
        """Strategic fragment proposal based on negotiation history."""
        if not self.fragments:
            return [TextFragment("negotiator_initial")]
        
        # Analyze successful treaties
        if self.successful_treaties:
            best_pattern = max(self.successful_treaties.items(),
                             key=lambda x: x[1])[0]
            return [TextFragment(f"treaty_proposal_{best_pattern}")]
        
        # Score existing fragments
        scores = self._score_fragments()
        best_idx = scores.index(max(scores))
        best_fragment = self.fragments[best_idx]
        
        # Propose new treaty based on best fragment
        if isinstance(best_fragment, TreatyFragment):
            # Build upon existing treaty
            new_terms = best_fragment.terms.copy()
            new_terms.append(f"extension_{random.randint(1,100)}")
            new_treaty = TreatyFragment(f"{best_fragment.data}_extended",
                                      best_fragment.parties.copy())
            new_treaty.add_terms(new_terms)
            return [new_treaty]
        else:
            # Convert fragment to treaty proposal
            return [TreatyFragment(f"proposal_{best_fragment.data}",
                                 {self.id})]

    def refine_fragments(self, environment_feedback: Dict, ecology: 'ConceptualEcology'):
        """Refine fragments with focus on treaty formation."""
        consistency = environment_feedback.get("consistency_score", 0)
        
        if consistency < self.negotiation_threshold:
            # Remove unsuccessful fragments
            if self.fragments:
                removed = self.fragments.pop(random.randint(0, len(self.fragments)-1))
                if isinstance(removed, TreatyFragment):
                    # Record failed treaty pattern
                    pattern = "_".join(removed.terms)
                    self.successful_treaties[pattern] -= 0.5
            self.energy -= 10
        else:
            # Reinforce successful patterns
            for fragment in self.fragments:
                if isinstance(fragment, TreatyFragment):
                    pattern = "_".join(fragment.terms)
                    self.successful_treaties[pattern] += consistency
            self.energy += 8
        
        # Update treaty-related fragments
        for fragment in self.fragments:
            if ecology.treaty_store.is_stable(fragment):
                fragment.update_quality(5)
                if isinstance(fragment, TreatyFragment):
                    pattern = "_".join(fragment.terms)
                    self.successful_treaties[pattern] += 2

class ExplorerAgent(ConceptualAgent):
    """Enhanced explorer agent with improved exploration strategies."""
    def __init__(self, 
                 name: Optional[str] = None,
                 specialization: AgentSpecialization = AgentSpecialization.GENERAL,
                 exploration_rate: float = 0.3):
        super().__init__(
            name=name,
            specialization=specialization,
            agent_type=AgentType.EXPLORER
        )
        self.exploration_rate = exploration_rate
        self.novelty_threshold = 0.2
        self.exploration_history: List[Dict] = []
        self.successful_patterns: Dict[str, int] = defaultdict(int)

    def propose_fragments(self, ecology: 'ConceptualEcology') -> List[ConceptualFragment]:
        """Generate new fragments with intelligent exploration."""
        # Decide whether to explore or exploit
        if random.random() < self.exploration_rate:
            # Exploration: Generate completely new patterns
            return self._generate_novel_fragments()
        else:
            # Exploitation: Build upon successful patterns
            return self._build_on_success()

    def _generate_novel_fragments(self) -> List[ConceptualFragment]:
        """Generate novel fragments using various strategies."""
        if random.random() < 0.5:
            # Generate text fragment
            words = ["explore", "discover", "novel", "pattern"]
            new_text = f"{random.choice(words)}_{random.randint(1,100)}"
            return [TextFragment(new_text)]
        else:
            # Generate vector fragment
            dimension = random.randint(2, 5)
            vector_data = [random.uniform(-1, 1) for _ in range(dimension)]
            return [VectorFragment(vector_data)]

    def _extract_keywords(self) -> Set[str]:
        """Extract important keywords from text."""
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
        words = self.data.lower().split()
        return {word for word in words if word not in stopwords}

    def similarity(self, other: ConceptualFragment) -> float:
        if not isinstance(other, TextFragment):
            return 0.0
        
        # Combine embedding similarity with other metrics
        embedding_sim = np.dot(self.embedding, other.embedding) / (
            np.linalg.norm(self.embedding) * np.linalg.norm(other.embedding))
        
        # Keyword overlap
        keywords_a = self.keywords
        keywords_b = other.keywords
        keyword_sim = len(keywords_a & keywords_b) / max(1, len(keywords_a | keywords_b))
        
        # Sentiment similarity
        sentiment_sim = 1.0 - abs(self.sentiment_score - other.sentiment_score) / 2.0
        
        # Weighted combination
        return 0.6 * embedding_sim + 0.2 * keyword_sim + 0.2 * sentiment_sim

    def merge(self, other: ConceptualFragment) -> ConceptualFragment:
        if not isinstance(other, TextFragment):
            raise TypeError("Can only merge with TextFragments")
        
        # More sophisticated merging strategy
        merged_text = f"{self.data}_{other.data}"
        merged = TextFragment(merged_text)
        
        # Combine qualities and metrics
        merged.quality = (self.quality + other.quality) / 2
        merged.tags = self.tags.union(other.tags)
        
        # Record merger in history
        merged.modification_history.extend(self.modification_history)
        merged.modification_history.extend(other.modification_history)
        merged.modification_history.append({
            'timestamp': datetime.now(),
            'type': 'merge',
            'parents': [str(self.id), str(other.id)]
        })
        
        return merged

class MetaAgent:
    """Monitors and regulates the ecology."""
    def __init__(self):
        self.interventions = 0
        self.observation_history: List[Dict] = []
        self.adaptation_threshold = 0.3

    def monitor_and_act(self, ecology: 'ConceptualEcology'):
        """Monitor ecology state and intervene if necessary."""
        # Analyze population diversity
        agent_types = defaultdict(int)
        for agent in ecology.agents:
            agent_types[agent.agent_type] += 1
        
        # Calculate metrics
        avg_fitness = (sum(a.fitness for a in ecology.agents) / 
                      len(ecology.agents) if ecology.agents else 0)
        
        type_diversity = len(agent_types) / len(AgentType)
        
        # Record observations
        self.observation_history.append({
            'generation': ecology.generation,
            'avg_fitness': avg_fitness,
            'type_diversity': type_diversity,
            'agent_counts': dict(agent_types)
        })
        
        # Decide on interventions
        if ecology.generation % 5 == 0 and avg_fitness < 500:
            if type_diversity < self.adaptation_threshold:
                self._add_new_agent_type(ecology)
            if len(ecology.environments) < len(EnvironmentType):
                self._add_new_environment(ecology)

    def _add_new_agent_type(self, ecology: 'ConceptualEcology'):
        """Add underrepresented agent type."""
        agent_counts = defaultdict(int)
        for agent in ecology.agents:
            agent_counts[agent.agent_type] += 1
        
        # Find least represented type
        all_types = set(AgentType)
        existing_types = set(agent_counts.keys())
        missing_types = all_types - existing_types
        
        if missing_types:
            new_type = random.choice(list(missing_types))
            new_agent = self._create_agent(new_type)
            ecology.add_agent(new_agent)
            self.interventions += 1

    def _add_new_environment(self, ecology: 'ConceptualEcology'):
        """Add new environment type."""
        existing_types = {env.env_type for env in ecology.environments}
        missing_types = set(EnvironmentType) - existing_types
        
        if missing_types:
            new_type = random.choice(list(missing_types))
            new_env = self._create_environment(new_type)
            ecology.add_environment(new_env)
            self.interventions += 1

    def _create_agent(self, agent_type: AgentType) -> ConceptualAgent:
        """Create new agent of specified type."""
        if agent_type == AgentType.EXPLORER:
            return ExplorerAgent()
        elif agent_type == AgentType.CONSERVATIVE:
            return ConservativeAgent()
        elif agent_type == AgentType.NEGOTIATOR:
            return NegotiatorAgent()
        return ConceptualAgent()

    def _create_environment(self, env_type: EnvironmentType) -> EnhancedEnvironment:
        """Create new environment of specified type."""
        if env_type == EnvironmentType.LOGICAL:
            return LogicalEnvironment()
        elif env_type == EnvironmentType.PROBLEM_SOLVING:
            return ProblemSolvingEnvironment()
        elif env_type == EnvironmentType.SOCIAL:
            return SocialEnvironment()
        return EnhancedEnvironment()
