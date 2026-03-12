import threading
import logging

logger = logging.getLogger(__name__)

class ThreadManager:
    """
    Manages multiple threads used to execute given handler tasks.
    """

    def __init__(self, handlers):
        self.handlers = handlers
        self.threads = []

    def start(self):
        for handler in self.handlers:
            thread = threading.Thread(target=handler.run, name=handler.__class__.__name__)
            thread.daemon = False  # Ensure threads are waited for on shutdown
            self.threads.append(thread)
            thread.start()

    def stop(self):
        # Signal all handlers to stop
        logger.info("ThreadManager: Triggering stop event for all handlers.")
        for handler in self.handlers:
            handler.stop_event.set()

        # Wait for all threads to finish with timeout
        for i, thread in enumerate(self.threads):
            if thread.is_alive():
                logger.info(f"ThreadManager: Waiting for {thread.name} to finish...")
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning(f"ThreadManager: Thread {thread.name} did not terminate within timeout.")
