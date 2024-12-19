from copy import copy
from random import random
import uuid
from step import Strategy
from typing import List, Dict, Optional
from collections import defaultdict
from agents import ConceptualAgent, MetaAgent
from environments import EnhancedEnvironment
from stores import TreatyStore
from processors import ParallelProcessor

class ConceptualEcology:
    """Enhanced CCE system with parallel processing and advanced evolution."""
    
    def __init__(self, num_threads: int = 4):
        self.agents: List[ConceptualAgent] = []
        self.environments: List[EnhancedEnvironment] = []
        self.generation = 0
        self.history: List[Dict] = []
        self.treaty_store = TreatyStore()
        self.meta_agent = MetaAgent()
        
        # Parallel processing support
        self.parallel_processor = ParallelProcessor(num_threads)
        self.parallel_processor.start()
        
        # Enhanced tracking
        self.population_stats: Dict[str, List[float]] = defaultdict(list)
        self.environment_stats: Dict[str, List[float]] = defaultdict(list)
        self.treaty_stats: Dict[str, List[int]] = defaultdict(list)
        
        # Evolution parameters
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.selection_pressure = 0.8
        self.diversity_threshold = 0.3
        
    def __del__(self):
        """Cleanup parallel processing resources."""
        if hasattr(self, 'parallel_processor'):
            self.parallel_processor.stop()

    def add_agent(self, agent: ConceptualAgent):
        """Add new agent to the ecology."""
        agent.generation = self.generation
        self.agents.append(agent)

    def add_environment(self, env: EnhancedEnvironment):
        """Add new environment to the ecology."""
        self.environments.append(env)

    def step(self):
        """Enhanced evolutionary step with parallel processing."""
        self.generation += 1
        
        # 1. Parallel agent proposals
        for agent in self.agents:
            new_fragments = agent.propose_fragments()
            agent.fragments.extend(new_fragments)
        
        # 2. Parallel agent interactions
        self.parallel_processor.process_interactions(self.agents, self)
        
        # 3. Parallel environmental evaluation
        evaluation_results = self.parallel_processor.process_evaluations(
            self.agents, self.environments, self
        )
        
        # Process evaluation results
        for agent, results in zip(self.agents, evaluation_results):
            # Aggregate feedback across environments
            feedback = defaultdict(list)
            for result in results:
                for k, v in result.items():
                    feedback[k].append(v)
            
            # Average scores
            aggregated_feedback = {
                k: sum(v) / len(v) for k, v in feedback.items()
            }
            
            # Update agent
            agent.refine_fragments(aggregated_feedback, self)
            agent.fitness = agent.get_fitness(aggregated_feedback)
        
        # 4. Enhanced evolution
        self._evolve_population()
        
        # 5. Update treaty store
        self.treaty_store.decay_treaties(self.agents)
        
        # 6. Meta-agent monitoring
        self.meta_agent.monitor_and_act(self)
        
        # 7. Record statistics
        self._record_statistics()

    def _evolve_population(self):
        """Enhanced population evolution with diversity preservation."""
        if not self.agents:
            return
        
        # Sort by fitness
        self.agents.sort(key=lambda a: a.fitness, reverse=True)
        
        # Calculate population diversity
        diversity = self._calculate_diversity()
        
        # Determine survival cutoff based on diversity
        if diversity < self.diversity_threshold:
            # Preserve more of population if diversity is low
            survival_cutoff = int(len(self.agents) * 0.7)
        else:
            survival_cutoff = len(self.agents) // 2
        
        # Select survivors
        survivors = self.agents[:survival_cutoff]
        
        # Generate offspring
        new_agents = []
        while len(new_agents) < len(self.agents) - survival_cutoff:
            if random.random() < self.crossover_rate and len(survivors) >= 2:
                # Crossover
                parent1, parent2 = random.sample(survivors, 2)
                child = self._crossover(parent1, parent2)
            else:
                # Cloning with mutation
                parent = random.choice(survivors)
                child = parent.reproduce()
                if child and random.random() < self.mutation_rate:
                    child.mutate()
            
            if child:
                new_agents.append(child)
        
        # Update population
        self.agents = survivors + new_agents

    def _calculate_diversity(self) -> float:
        """Calculate population diversity based on fragment and strategy differences."""
        if not self.agents or len(self.agents) < 2:
            return 0.0
        
        diversity_scores = []
        
        # Compare each agent pair
        for i, agent1 in enumerate(self.agents):
            for agent2 in self.agents[i+1:]:
                # Strategy diversity
                strategy_diff = sum(
                    abs(agent1.strategy_preferences[s] - agent2.strategy_preferences[s])
                    for s in Strategy
                ) / len(Strategy)
                
                # Fragment diversity
                fragment_sim = 0.0
                if agent1.fragments and agent2.fragments:
                    similarities = [
                        max(f1.similarity(f2) for f2 in agent2.fragments)
                        for f1 in agent1.fragments
                    ]
                    fragment_sim = sum(similarities) / len(similarities)
                
                # Combined diversity score
                pair_diversity = (1 - fragment_sim + strategy_diff) / 2
                diversity_scores.append(pair_diversity)
        
        return sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0

    def _crossover(self, parent1: ConceptualAgent, parent2: ConceptualAgent) -> Optional[ConceptualAgent]:
        """Perform crossover between two parent agents."""
        if parent1.energy < 60 or parent2.energy < 60:
            return None
        
        # Create child with mixed properties
        child = copy.deepcopy(parent1)
        child.id = uuid.uuid4()
        child.name = f"Agent-{child.id}"
        child.generation = self.generation
        
        # Mix strategy preferences
        crossover_point = random.randint(0, len(Strategy) - 1)
        strategies = list(Strategy)
        for i, strategy in enumerate(strategies):
            if i < crossover_point:
                child.strategy_preferences[strategy] = parent1.strategy_preferences[strategy]
            else:
                child.strategy_preferences[strategy] = parent2.strategy_preferences[strategy]