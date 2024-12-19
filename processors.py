from typing import List
import queue
import threading
from agents import ConceptualAgent
from ecologies import ConceptualEcology
from environments import VirtualEnvironment

class ParallelProcessor:
    """Handles parallel processing of agent interactions and evaluations."""
    
    def __init__(self, num_threads: int = 4):
        self.num_threads = num_threads
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers: List[threading.Thread] = []
        self.running = False

    def start(self):
        """Start worker threads."""
        self.running = True
        for _ in range(self.num_threads):
            worker = threading.Thread(target=self._worker_loop)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def stop(self):
        """Stop worker threads."""
        self.running = False
        for worker in self.workers:
            worker.join()
        self.workers.clear()

    def _worker_loop(self):
        """Main worker thread loop."""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break
                    
                func, args = task
                result = func(*args)
                self.result_queue.put(result)
                self.task_queue.task_done()
            except queue.Empty:
                continue

    def process_interactions(self, agents: List[ConceptualAgent], ecology: 'ConceptualEcology'):
        """Process agent interactions in parallel."""
        for i, agent in enumerate(agents):
            others = agents[:i] + agents[i+1:]
            self.task_queue.put((agent.interact_with_others, (others, ecology)))
        
        self.task_queue.join()
        while not self.result_queue.empty():
            self.result_queue.get()

    def process_evaluations(self, agents: List[ConceptualAgent], 
                          environments: List[VirtualEnvironment],
                          ecology: 'ConceptualEcology'):
        """Process environmental evaluations in parallel."""
        for agent in agents:
            for env in environments:
                self.task_queue.put((env.evaluate, (agent.fragments, ecology)))
        
        results = []
        self.task_queue.join()
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results
