from typing import List, Dict
from collections import defaultdict
from datetime import datetime
from conceptual_fragment import TreatyFragment
from agents import ConceptualAgent
from fragments import ConceptualFragment

class TreatyStore:
    """Manages stable conceptual treaties formed by agents."""
    def __init__(self):
        self.treaties = defaultdict(int)
        self.stable_treaties = set()
        self.validation_history: List[Dict] = []

    def record_treaty(self, fragment: TreatyFragment):
        """Record a new treaty or reinforce existing one."""
        self.treaties[fragment.data] += 1
        if self.treaties[fragment.data] > 3:
            self.stable_treaties.add(fragment.data)
            self.validation_history.append({
                'timestamp': datetime.now(),
                'treaty': fragment.data,
                'stability_score': fragment.stability_score
            })

    def is_stable(self, fragment: ConceptualFragment) -> bool:
        """Check if a fragment represents a stable treaty."""
        if isinstance(fragment, TreatyFragment):
            return fragment.data in self.stable_treaties
        return False

    def validate_treaty(self, treaty: TreatyFragment, agents: List[ConceptualAgent]) -> bool:
        """Validate if the treaty is still relevant."""
        if not agents:
            return False
        
        agreement_count = sum(
            1 for agent in agents
            if any(treaty.similarity(f) > 0.9 for f in agent.fragments)
        )
        
        return agreement_count / len(agents) > 0.6

    def decay_treaties(self, agents: List[ConceptualAgent]):
        """Remove treaties that are no longer validated."""
        unstable_treaties = set()
        for treaty_data in self.stable_treaties:
            treaty = TreatyFragment(treaty_data)
            if not self.validate_treaty(treaty, agents):
                unstable_treaties.add(treaty_data)
                self.treaties[treaty_data] = 0
        
        self.stable_treaties -= unstable_treaties
