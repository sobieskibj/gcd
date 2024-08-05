import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

class Strategy(ABC):

    def run(self):
        log.info("Running")
        log.info("Starting pre main loop")
        self.pre_main_loop()
        log.info("Finished pre main loop")
        log.info("Starting main loop")
        for iter in range(self.n_iters):
            log.info(f"Iteration: {iter}")
            log.info(f"Starting pre DAE loop")
            self.pre_dae_loop(iter)
            log.info(f"Finished pre DAE loop")
            log.info(f"Starting DAE loop")
            for step_idx in range(self.n_steps_dae):
                self.dae_step(step_idx, iter)
            log.info(f"Finished DAE loop")
            log.info(f"Starting post DAE loop")
            self.post_dae_loop(iter)
            log.info(f"Finished post DAE loop")
            log.info(f"Starting pre proxy loop")
            self.pre_proxy_loop(iter)
            log.info(f"Finished pre proxy loop")
            log.info(f"Starting proxy loop")
            for step_idx in range(self.n_steps_proxy):
                self.proxy_step(step_idx, iter)
            log.info(f"Finished proxy loop")
            log.info(f"Starting post proxy loop")
            self.post_proxy_loop(iter)
            log.info(f"Finished post proxy loop")
        log.info(f"Finished main loop")
        log.info(f"Starting post main loop")
        self.post_main_loop()
        log.info(f"Finished post main loop")
        log.info(f"Finished")

    @abstractmethod
    def pre_main_loop(self):
        pass

    @abstractmethod
    def pre_dae_loop(self):
        pass
    
    @abstractmethod
    def dae_step(self, step_idx):
        pass

    @abstractmethod
    def post_dae_loop(self):
        pass

    @abstractmethod
    def pre_proxy_loop(self):
        pass

    @abstractmethod
    def proxy_step(self, step_idx):
        pass
    
    @abstractmethod
    def post_proxy_loop(self):
        pass
        
    @abstractmethod
    def post_main_loop(self):
        pass