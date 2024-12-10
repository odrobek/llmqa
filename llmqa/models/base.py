from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for LLM models.
    
    All model implementations should inherit from this class and implement
    the required methods.
    """
    
    @abstractmethod
    def __init__(self):
        """Initialize the model."""
        pass
    
    @abstractmethod
    def __call__(self, message: str) -> str:
        """Process a message and return a response.
        
        Args:
            message (str): The input message to process
            
        Returns:
            str: The model's response
        """
        pass 