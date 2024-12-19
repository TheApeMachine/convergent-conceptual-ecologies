from abc import ABC, abstractmethod
import uuid
import random
from typing import List, Dict, Any, Optional, Tuple
import copy
import math
from enum import Enum
from collections import defaultdict
import numpy as np

##########################################
# Basic Data Structures and Interfaces
##########################################

class ConceptualFragment(ABC):
    """
    Abstract class for conceptual fragments.
    """
    def __init__(self, data: Any):
        self.data = data
        self.quality = 0.0
        self.usage_count = 0 # new fragment quality metric
        self.durability = 0 # fragment durability
        self.creation_generation = 0
    @abstractmethod
    def similarity(self, other: 'ConceptualFragment') -> float:
        pass

    @abstractmethod
    def merge(self, other: 'ConceptualFragment') -> 'ConceptualFragment':
        pass

    def update_quality(self, score: float):
       # quality goes up if similar fragments appear
       self.quality += score

    def on_usage(self):
      self.usage_count += 1
      self.durability +=1
    
    def reset_durability(self):
        self.durability = 0

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data}, q={self.quality:.2f})"


word_embeddings = {}  # global embedding dictionary

class TextFragment(ConceptualFragment):
    """Text fragment with word embedding and similarity."""

    def __init__(self, data, embedding_dim=5):
        super().__init__(data)
        self.embedding = self._compute_embedding(data, embedding_dim)

    def _compute_embedding(self, text, embedding_dim):
        # Simple averaging of token embeddings (replace with a true embedding model)
        tokens = text.lower().split()
        embeddings = []
        for token in tokens:
            if token in word_embeddings:
                embeddings.append(word_embeddings[token])
            else: # new word, generate a new embedding
                new_embedding = [random.uniform(-0.2, 0.2) for _ in range(embedding_dim)]
                word_embeddings[token] = new_embedding
                embeddings.append(new_embedding)
        
        if embeddings:
            avg_embedding = [sum(emb[i] for emb in embeddings) / len(embeddings) for i in range(embedding_dim)]
            return avg_embedding
        else:
            return [0] * embedding_dim  # zero embedding for empty string

    def similarity(self, other: ConceptualFragment) -> float:
        if not isinstance(other, TextFragment):
            return 0.0
        # cosine similarity of embeddings
        v1 = np.array(self.embedding)
        v2 = np.array(other.embedding)

        dot_product = np.dot(v1,v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
           return 0.0

        return dot_product / (norm1 * norm2)

    def merge(self, other: ConceptualFragment) -> ConceptualFragment:
        if not isinstance(other, TextFragment):
            raise TypeError("Can only merge with TextFragments")
        merged_text = self.data + "_" + other.data
        return TextFragment(merged_text)


class VectorFragment(ConceptualFragment):
    """Vector-based conceptual fragment."""
    def __init__(self, data):
        super().__init__(data)
        if not isinstance(data, list):
            raise TypeError("Vector data must be a list")
        self.dimensionality = len(data)

    def similarity(self, other: ConceptualFragment) -> float:
        if not isinstance(other, VectorFragment) or self.dimensionality != other.dimensionality:
            return 0.0
        v1 = np.array(self.data)
        v2 = np.array(other.data)

        dot_product = np.dot(v1,v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
          return 0.0
        return dot_product / (norm1 * norm2)

    def merge(self, other: ConceptualFragment) -> ConceptualFragment:
        if not isinstance(other, VectorFragment) or self.dimensionality != other.dimensionality:
            raise TypeError("Vectors must be the same size for merge")
        merged_data = [x+y for x,y in zip(self.data, other.data)]
        return VectorFragment(merged_data)

class TreatyFragment(ConceptualFragment):
    """Treaty fragment, with quality metrics."""
    def __init__(self, data):
        super().__init__(data)

    def similarity(self, other: 'ConceptualFragment') -> float:
        if isinstance(other, TreatyFragment) or isinstance(other, TextFragment):
            return 1.0 if self.data == other.data else 0.0
        return 0.0

    def merge(self, other: 'ConceptualFragment') -> 'ConceptualFragment':
        if isinstance(other, TreatyFragment) or isinstance(other, TextFragment):
           return TreatyFragment(self.data + "&" + other.data)
        raise TypeError("Can only merge Treaties with Treaties or TextFragments")

    def __repr__(self):
        return f"TreatyFragment({self.data}, q={self.quality:.2f})"

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


##########################################
# Agents
##########################################

class ConceptualAgent(ABC):
    """Abstract base class for conceptual agents."""
    def __init__(self, name: Optional[str] = None, specialization: AgentSpecialization = AgentSpecialization.GENERAL, agent_type: AgentType = AgentType.EXPLORER):
        self.id = uuid.uuid4()
        self.name = name if name else f"Agent-{self.id}"
        self.specialization = specialization
        self.agent_type = agent_type
        self.fragments: List[ConceptualFragment] = []
        self.energy = 100.0
        self.health = 100.0
        self.fitness = 1.0
        self.interaction_memory = defaultdict(int)
        self.lineage = [self.id]

    @abstractmethod
    def propose_fragments(self) -> List[ConceptualFragment]:
        pass

    @abstractmethod
    def refine_fragments(self, environment_feedback: Dict, ecology: 'ConceptualEcology'):
        pass

    @abstractmethod
    def interact_with_others(self, others: List['ConceptualAgent'], ecology: 'ConceptualEcology'):
        pass

    def mutate(self):
        if self.fragments:
            idx = random.randint(0, len(self.fragments)-1)
            old_fragment = self.fragments[idx]
            if isinstance(old_fragment, TextFragment):
                mutated_data = f"{old_fragment.data}_mutated"
                self.fragments[idx] = TextFragment(mutated_data)
            elif isinstance(old_fragment, VectorFragment):
                mutated_data = [x + random.uniform(-0.1, 0.1) for x in old_fragment.data]
                self.fragments[idx] = VectorFragment(mutated_data)
            self.energy -= 5.0

    def reproduce(self) -> Optional['ConceptualAgent']:
        if self.energy > 120:
            child = copy.deepcopy(self)
            child.id = uuid.uuid4()
            child.name = f"Agent-{child.id}"
            child.energy = 100.0
            child.lineage = self.lineage + [child.id]
            child.mutate()
            self.energy -= 30
            return child
        return None

    def get_fitness(self, env_feedback: Dict) -> float:
       # combining the environment score, with agent internal health and energy
       consistency_score = env_feedback.get("consistency_score", 0)
       return self.health * self.energy * (consistency_score + 0.1)

    def _score_fragments(self) -> List[float]:
        scores = []
        for f in self.fragments:
            if isinstance(f, VectorFragment):
                scores.append(sum(x*x for x in f.data))
            elif isinstance(f, TextFragment):
                scores.append(f.quality) # quality influences scoring
            elif isinstance(f, TreatyFragment):
                scores.append(f.quality + 10) # treaties are highly valued
            else:
                scores.append(0)
        return scores

    def __repr__(self):
        return f"<ConceptualAgent {self.name} | type={self.agent_type.value} spec={self.specialization.value} | fragments={len(self.fragments)}>"

class BasicConceptualAgent(ConceptualAgent):
    def __init__(self, name: Optional[str] = None, specialization: AgentSpecialization = AgentSpecialization.GENERAL):
        super().__init__(name=name, specialization=specialization, agent_type=AgentType.BASIC)

    def propose_fragments(self) -> List[ConceptualFragment]:
        if not self.fragments:
            return [TextFragment("basic_hypothesis")]
        else:
            combined_data = "_AND_".join([f.data for f in self.fragments if isinstance(f, TextFragment)])
            return [TextFragment(f"combined({combined_data})")]

    def refine_fragments(self, environment_feedback: Dict, ecology: 'ConceptualEcology'):
        if environment_feedback.get("consistency_score", 1) < 0.5:
            if self.fragments:
                self.fragments.pop(random.randint(0, len(self.fragments)-1))
            self.energy -= 10
        else:
            self.energy += 5

        for f in self.fragments:
            if ecology.treaty_store.is_stable(f):
                f.update_quality(2)

    def interact_with_others(self, others: List[ConceptualAgent], ecology: 'ConceptualEcology'):
        if not self.fragments:
            return
        partner = random.choice(others) if others else None
        if partner and partner.fragments:
            sim = self.fragments[0].similarity(partner.fragments[0])
            if sim > 0.9:
                # record a treaty
                treaty = TreatyFragment(self.fragments[0].data + "&" + partner.fragments[0].data)
                ecology.record_treaty(treaty)
                self.fragments.append(treaty)
            else:
                treaty_data = f"treaty({self.fragments[0].data}, {partner.fragments[0].data})"
                self.fragments.append(TextFragment(treaty_data))
            self.energy += 2
            self.interaction_memory[partner.id] += 1
            for f in self.fragments:
                f.on_usage()


class ExplorerAgent(ConceptualAgent):
    def __init__(self, name: Optional[str] = None, specialization: AgentSpecialization = AgentSpecialization.GENERAL):
        super().__init__(name=name, specialization=specialization, agent_type=AgentType.EXPLORER)

    def propose_fragments(self) -> List[ConceptualFragment]:
        rand_str = "".join([chr(random.randint(97, 122)) for _ in range(random.randint(5,15))])
        rand_vector = [random.uniform(-1,1) for _ in range(5)]
        return [TextFragment(f"explorer-{rand_str}"), VectorFragment(rand_vector)]

    def refine_fragments(self, environment_feedback: Dict, ecology: 'ConceptualEcology'):
        if environment_feedback.get("consistency_score", 1) < 0.2:
            self.energy -= 2
        else:
            self.energy += 8
        
        for f in self.fragments:
             if ecology.treaty_store.is_stable(f):
                f.update_quality(2)

    def interact_with_others(self, others: List[ConceptualAgent], ecology: 'ConceptualEcology'):
        if not self.fragments:
            return
        partner = random.choice(others) if others else None
        if partner and partner.fragments:
            try:
                f1 = random.choice(self.fragments)
                f2 = random.choice(partner.fragments)
                merged = f1.merge(f2)
                sim = f1.similarity(f2)
                self.fragments.append(merged)
                self.energy +=3
                if sim > 0.9:
                   treaty = TreatyFragment(merged.data)
                   ecology.record_treaty(treaty)
            except TypeError:
               pass
            self.interaction_memory[partner.id] += 1
            for f in self.fragments:
                f.on_usage()

class ConservativeAgent(ConceptualAgent):
    def __init__(self, name: Optional[str] = None, specialization: AgentSpecialization = AgentSpecialization.LOGICAL):
        super().__init__(name=name, specialization=specialization, agent_type=AgentType.CONSERVATIVE)

    def propose_fragments(self) -> List[ConceptualFragment]:
        if not self.fragments:
            return [TextFragment("conservative_initial")]
        
        scores = self._score_fragments()
        best_frag_idx = scores.index(max(scores))
        best_frag = self.fragments[best_frag_idx]

        if isinstance(best_frag, TextFragment):
            new_data = best_frag.data + "_refined"
            return [TextFragment(new_data)]
        elif isinstance(best_frag, VectorFragment):
            new_data = [x+random.uniform(-0.1, 0.1) for x in best_frag.data]
            return [VectorFragment(new_data)]
        elif isinstance(best_frag, TreatyFragment):
            return [best_frag] # stable treaties are promoted
        return []

    def refine_fragments(self, environment_feedback: Dict, ecology: 'ConceptualEcology'):
        if environment_feedback.get("consistency_score", 1) < 0.4:
           if self.fragments:
             scores = self._score_fragments()
             worst_frag_idx = scores.index(min(scores))
             self.fragments.pop(worst_frag_idx)
           self.energy -= 5
        else:
           self.energy += 10
        for f in self.fragments:
            if ecology.treaty_store.is_stable(f):
               f.update_quality(3)

    def interact_with_others(self, others: List[ConceptualAgent], ecology: 'ConceptualEcology'):
       if not self.fragments:
         return
       eligible_partners = [a for a in others if a.specialization == self.specialization]
       partner = random.choice(eligible_partners) if eligible_partners else (random.choice(others) if others else None)

       if partner and partner.fragments:
            my_scores = self._score_fragments()
            partner_scores = partner._score_fragments()

            if my_scores and partner_scores:
                best_my_idx = my_scores.index(max(my_scores))
                best_partner_idx = partner_scores.index(max(partner_scores))
                
                best_my_frag = self.fragments[best_my_idx]
                best_partner_frag = partner.fragments[best_partner_idx]
                
                if best_my_frag.similarity(best_partner_frag) > 0.5:
                  try:
                    merged = best_my_frag.merge(best_partner_frag)
                    self.fragments.append(merged)
                    self.energy +=5
                    if best_my_frag.similarity(best_partner_frag) > 0.9:
                        treaty = TreatyFragment(merged.data)
                        ecology.record_treaty(treaty)
                  except TypeError:
                    pass
            self.interaction_memory[partner.id] += 1
            for f in self.fragments:
                f.on_usage()

class NegotiatorAgent(ConceptualAgent):
    def __init__(self, name: Optional[str] = None, specialization: AgentSpecialization = AgentSpecialization.SOCIAL):
        super().__init__(name=name, specialization=specialization, agent_type=AgentType.NEGOTIATOR)

    def propose_fragments(self) -> List[ConceptualFragment]:
        if not self.fragments:
           return [TextFragment("negotiator_initial")]
      
        scores = self._score_fragments()
        best_frag_idx = scores.index(max(scores))
        best_frag = self.fragments[best_frag_idx]
        return [best_frag]

    def refine_fragments(self, environment_feedback: Dict, ecology: 'ConceptualEcology'):
        if environment_feedback.get("consistency_score", 1) < 0.6:
            if self.fragments:
              self.fragments.pop(random.randint(0, len(self.fragments)-1))
            self.energy -= 8
        else:
          self.energy += 5
        for f in self.fragments:
            if ecology.treaty_store.is_stable(f):
                f.update_quality(5)

    def interact_with_others(self, others: List[ConceptualAgent], ecology: 'ConceptualEcology'):
        if not self.fragments:
            return

        eligible_partners = [a for a in others if a.agent_type != self.agent_type]
        partner = random.choice(eligible_partners) if eligible_partners else (random.choice(others) if others else None)

        if partner and partner.fragments:
            my_scores = self._score_fragments()
            partner_scores = partner._score_fragments()

            if my_scores and partner_scores:
                best_my_idx = my_scores.index(max(my_scores))
                best_partner_idx = partner_scores.index(max(partner_scores))

                best_my_frag = self.fragments[best_my_idx]
                best_partner_frag = partner.fragments[best_partner_idx]

                similarity = best_my_frag.similarity(best_partner_frag)

                if similarity > 0.7:
                    try:
                        merged = best_my_frag.merge(best_partner_frag)
                        self.fragments.append(merged)
                        self.energy += 10
                        if similarity > 0.9:
                           treaty = TreatyFragment(merged.data)
                           ecology.record_treaty(treaty)
                    except TypeError:
                        pass
            self.interaction_memory[partner.id] += 1
            for f in self.fragments:
                f.on_usage()


##########################################
# Treaty Store
##########################################

class TreatyStore:
    """Manages stable conceptual treaties formed by agents."""
    def __init__(self):
        self.treaties = defaultdict(int)
        self.stable_treaties = set()

    def record_treaty(self, fragment: TreatyFragment):
        self.treaties[fragment.data] += 1
        if self.treaties[fragment.data] > 3:
            self.stable_treaties.add(fragment.data)

    def is_stable(self, fragment: ConceptualFragment) -> bool:
        if isinstance(fragment, TreatyFragment):
            return fragment.data in self.stable_treaties
        return False

    def validate_treaty(self, treaty: TreatyFragment, agents: List[ConceptualAgent]) -> bool:
       """
       Validate if the treaty is still relevant, by making sure most agents
       agree with it.
       """
       
       agreement_count = 0
       for agent in agents:
           for frag in agent.fragments:
              if treaty.similarity(frag) > 0.9:
                 agreement_count += 1
       
       return agreement_count / len(agents) > 0.6 if agents else False


    def decay_treaties(self, agents: List[ConceptualAgent]):
        """Remove treaties that are no longer validated, making room for better ideas"""
        unstable_treaties = set()
        for treaty_data in self.stable_treaties:
            treaty = TreatyFragment(treaty_data)
            if not self.validate_treaty(treaty, agents):
                unstable_treaties.add(treaty_data)
        self.stable_treaties = self.stable_treaties.difference(unstable_treaties)
        # also reset the count for these treaties
        for t in unstable_treaties:
           self.treaties[t] = 0

class EnvironmentType(Enum):
    LOGICAL = "logical"
    PROBLEM_SOLVING = "problem_solving"
    SOCIAL = "social"


class VirtualEnvironment(ABC):
    def __init__(self, name: Optional[str] = None, env_type: EnvironmentType = EnvironmentType.LOGICAL, specialization: AgentSpecialization = AgentSpecialization.GENERAL):
        self.id = uuid.uuid4()
        self.name = name if name else f"Env-{self.id}"
        self.env_type = env_type
        self.specialization = specialization

    @abstractmethod
    def evaluate(self, fragments: List[ConceptualFragment], ecology: 'ConceptualEcology') -> Dict:
        pass

    def __repr__(self):
      return f"<VirtualEnvironment {self.name} | type={self.env_type.value} spec={self.specialization.value}>"


class LogicalEnvironment(VirtualEnvironment):
    def __init__(self, name: Optional[str] = None):
      super().__init__(name=name, env_type=EnvironmentType.LOGICAL, specialization=AgentSpecialization.LOGICAL)

    def evaluate(self, fragments: List[ConceptualFragment], ecology: 'ConceptualEcology') -> Dict:
        text_lengths = [len(f.data) for f in fragments if isinstance(f, TextFragment)]
        avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 1
        base_score = 1.0 / (1.0 + (avg_length / 50.0))

        # Bonus for referencing stable treaties
        stable_bonus = 0.0
        for f in fragments:
            if ecology.treaty_store.is_stable(f):
                stable_bonus += 0.3

        consistency_score = base_score + stable_bonus
        return {"consistency_score": min(consistency_score, 1.0)}


class ProblemSolvingEnvironment(VirtualEnvironment):
  def __init__(self, name: Optional[str] = None):
      super().__init__(name=name, env_type=EnvironmentType.PROBLEM_SOLVING, specialization=AgentSpecialization.PROBLEM_SOLVER)
      self.problem_vector = [random.uniform(-1, 1) for _ in range(5)]

  def evaluate(self, fragments: List[ConceptualFragment], ecology: 'ConceptualEcology') -> Dict:
    scores = []
    for f in fragments:
      if isinstance(f, VectorFragment):
        dot_product = sum(x*y for x, y in zip(f.data, self.problem_vector))
        magnitude = math.sqrt(sum(x*x for x in self.problem_vector))
        scores.append(dot_product / (magnitude if magnitude else 1))
    if not scores:
      return {"consistency_score": 0}
    avg_score = sum(scores)/len(scores)
    normalized_score = (avg_score + 1) / 2.0
    return {"consistency_score": normalized_score}


class SocialEnvironment(VirtualEnvironment):
  """Social environment rewards treaties and shared concepts."""
  def __init__(self, name: Optional[str] = None):
      super().__init__(name=name, env_type=EnvironmentType.SOCIAL, specialization=AgentSpecialization.SOCIAL)
  
  def evaluate(self, fragments: List[ConceptualFragment], ecology: 'ConceptualEcology') -> Dict:
     if not fragments or len(ecology.agents) <= 1:
        return {"consistency_score": 0}
     
     total_matches = 0
     count = 0
     # check if fragments align with treaties or appear in others
     for f in fragments:
         if ecology.treaty_store.is_stable(f):
             total_matches += 3  # huge bonus for stable treaties
             count += 1
         else:
              # check similarity with other agents
              for agent in ecology.agents:
                for frag in agent.fragments:
                   if f.similarity(frag) > 0.9 and agent.fragments is not fragments:
                     total_matches += 1
                     count += 1
     score = total_matches / (len(fragments) * len(ecology.agents)) if fragments else 0
     return {"consistency_score": min(score, 1.0)}


##########################################
# MetaAgent
##########################################

class MetaAgent:
    """
    A meta-agent that monitors the ecology and introduces changes over time.
    """
    def __init__(self):
        self.interventions = 0

    def monitor_and_act(self, ecology: 'ConceptualEcology'):
        # analyze diversity in the population
        agent_types = defaultdict(int)
        for a in ecology.agents:
           agent_types[a.agent_type] += 1
        
        agent_type_count = len(agent_types) # how many unique agent types exist?

        avg_fitness = sum(a.fitness for a in ecology.agents) / len(ecology.agents) if ecology.agents else 0
        if ecology.generation % 5 == 0 and avg_fitness < 500 and self.interventions < 2:
           #  add a new environment of a type we don't have yet
           #  favor types not in the population
            if agent_type_count < len(AgentType):
                for env_type in EnvironmentType:
                    if not any(e.env_type == env_type for e in ecology.environments):
                      new_env = self._create_new_environment(env_type, self.interventions)
                      ecology.add_environment(new_env)
                      self.interventions += 1
                      break

        # tune parameters
        if ecology.generation % 3 == 0:
            self.tune_parameters(ecology)
       

    def tune_parameters(self, ecology: 'ConceptualEcology'):
         # Check if treaties are stable
         if ecology.generation % 5 == 0:
             ecology.treaty_store.decay_treaties(ecology.agents)

         # check if population is balanced, introduce more new agents if necessary
         agent_types = defaultdict(int)
         for a in ecology.agents:
           agent_types[a.agent_type] += 1
         
         max_agent_type_count = max(agent_types.values()) if agent_types else 0
         if max_agent_type_count > 2 * (len(ecology.agents) / len(agent_types) if agent_types else 1):
            # add more of the least common type
             min_type = min(agent_types, key=agent_types.get) if agent_types else random.choice(list(AgentType))
             new_agent = self._create_new_agent(min_type, specialization=random.choice(list(AgentSpecialization)))
             ecology.add_agent(new_agent)
             print(f"MetaAgent: added new agent type: {min_type.value}")

    def _create_new_environment(self, env_type: EnvironmentType, intervention_num: int) -> VirtualEnvironment:
       if env_type == EnvironmentType.LOGICAL:
          return LogicalEnvironment(name=f"MetaLogicEnv_{intervention_num}")
       if env_type == EnvironmentType.PROBLEM_SOLVING:
           return ProblemSolvingEnvironment(name=f"MetaProblemEnv_{intervention_num}")
       if env_type == EnvironmentType.SOCIAL:
          return SocialEnvironment(name=f"MetaSocialEnv_{intervention_num}")
       raise ValueError("Unknown Environment type")


    def _create_new_agent(self, agent_type: AgentType, specialization: AgentSpecialization) -> ConceptualAgent:
        if agent_type == AgentType.EXPLORER:
            return ExplorerAgent(specialization=specialization)
        if agent_type == AgentType.CONSERVATIVE:
            return ConservativeAgent(specialization=specialization)
        if agent_type == AgentType.NEGOTIATOR:
           return NegotiatorAgent(specialization=specialization)
        return BasicConceptualAgent(specialization=specialization)

##########################################
# Conceptual Ecology
##########################################

class ConceptualEcology:
    """The entire CCE system."""
    def __init__(self):
        self.agents: List[ConceptualAgent] = []
        self.environments: List[VirtualEnvironment] = []
        self.generation = 0
        self.history = []
        self.treaty_store = TreatyStore()
        self.meta_agent = MetaAgent()

    def add_agent(self, agent: ConceptualAgent):
        self.agents.append(agent)

    def add_environment(self, env: VirtualEnvironment):
        self.environments.append(env)

    def record_treaty(self, fragment: TreatyFragment):
        self.treaty_store.record_treaty(fragment)

    def step(self):
        self.generation += 1

        # Agents propose
        for agent in self.agents:
            new_frags = agent.propose_fragments()
            agent.fragments.extend(new_frags)
        
        # Agents interact
        for agent in self.agents:
            others = [a for a in self.agents if a != agent]
            agent.interact_with_others(others, self)

        # Evaluate
        for agent in self.agents:
            feedback = {}
            for env in self.environments:
                env_fb = env.evaluate(agent.fragments, self)
                for k, v in env_fb.items():
                    feedback.setdefault(k, []).append(v)
            agg_feedback = {k: sum(vals)/len(vals) for k, vals in feedback.items()}
            agent.refine_fragments(agg_feedback, self)
            agent.fitness = agent.get_fitness(agg_feedback)

        # Evolution
        new_agents = []
        for agent in self.agents:
            agent.mutate()
            offspring = agent.reproduce()
            if offspring:
                new_agents.append(offspring)
            for frag in agent.fragments:
                frag.reset_durability() # reset all fragment durability
        self.agents.extend(new_agents)

        # Cull low fitness agents
        self.agents = [a for a in self.agents if a.energy > 0]
        self.agents = sorted(self.agents, key=lambda a: a.fitness, reverse=True)
        if len(self.agents) > 2:
            self.agents = self.agents[:len(self.agents)//2 + len(self.agents)%2]
        
        # Prune any long-lasting fragments in case they are no longer useful
        for agent in self.agents:
           agent.fragments = [f for f in agent.fragments if f.durability < 20]
        
        self._log_step()
        self.meta_agent.monitor_and_act(self)

    def run(self, steps=10):
        for i in range(steps):
            print(f"--- Generation {self.generation} ---")
            self.step()
            print(self.agents)

    def _log_step(self):
       agent_summary = defaultdict(int)
       for a in self.agents:
         agent_summary[a.agent_type] += 1

       env_summary = defaultdict(int)
       for e in self.environments:
         env_summary[e.env_type] += 1
         
       self.history.append(
          {
             "generation": self.generation,
             "agents": dict(agent_summary),
             "environments": dict(env_summary),
             "agent_fitnesses": [a.fitness for a in self.agents] if self.agents else [],
             "total_fragments": sum(len(a.fragments) for a in self.agents),
             "stable_treaties": len(self.treaty_store.stable_treaties),
             "total_embeddings": len(word_embeddings) # track how many new words are learned
          }
       )

    def print_stats(self):
       print("======================================")
       print("          SIMULATION SUMMARY")
       print("======================================")
       for item in self.history:
           avg_fitness = (sum(item['agent_fitnesses'])/len(item['agent_fitnesses'])) if item['agent_fitnesses'] else 0
           print(f"Gen: {item['generation']:>3} | Agents: {item['agents']} | Envs: {item['environments']} | Avg Fitness: {avg_fitness:.2f} | "
                 f"Total Fragments: {item['total_fragments']} | Stable Treaties: {item['stable_treaties']} | Total Embeddings: {item['total_embeddings']}")

##########################################
# Example Usage
##########################################

if __name__ == "__main__":
    ecology = ConceptualEcology()

    # Add initial agents of different types - using proper enum values
    ecology.add_agent(ExplorerAgent(specialization=AgentSpecialization.GENERAL))
    ecology.add_agent(ConservativeAgent(specialization=AgentSpecialization.LOGICAL))
    ecology.add_agent(NegotiatorAgent(specialization=AgentSpecialization.SOCIAL))
    ecology.add_agent(BasicConceptualAgent(specialization=AgentSpecialization.GENERAL))

    # Add initial environments
    ecology.add_environment(LogicalEnvironment(name="SimpleLogicEnv"))
    ecology.add_environment(ProblemSolvingEnvironment(name="VectorProblem"))
    ecology.add_environment(SocialEnvironment(name="SocialEnv"))

    # Run the ecology
    ecology.run(steps=100)
    ecology.print_stats()