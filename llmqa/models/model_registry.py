import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger('llmqa')

class ModelRegistry:
    """Central registry for managing LLM models across different providers."""
    
    def __init__(self):
        """Initialize the model registry."""
        self.models = {}  # All models in a single dictionary
        self.providers = {
            "databricks": {"display_name": "Databricks"},
            "google": {"display_name": "Google"},
            "rosie": {"display_name": "ROSIE"},
            "openrouter": {"display_name": "OpenRouter"},
            "groq": {"display_name": "Groq"}
        }
        self.qa_criteria = []  # Store QA criteria separately
        self.eval_criteria = []  # Store evaluation criteria separately
        self.config_path = self._get_config_path()
        self._cached_changes = {}  # Store pending changes
        self._has_pending_changes = False
        self.load_config()
        
    def _get_config_path(self) -> Path:
        """Get the path to the configuration file."""
        # Create ~/.llmqa directory if it doesn't exist
        config_dir = Path.home() / ".llmqa"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "config.json"
    
    def save_config(self):
        """Save the current configuration to the config file."""
        try:
            config = {
                "providers": self.providers,
                "models": self.models,
                "qa_criteria": self.qa_criteria,
                "eval_criteria": self.eval_criteria
            }
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.debug(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def load_config(self):
        """Load configuration from the config file if it exists."""
        if not self.config_path.exists():
            logger.info(f"Config file not found at {self.config_path}, using empty models list")
            return
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Update providers
            if "providers" in config:
                for provider_id, provider_data in config["providers"].items():
                    if provider_id in self.providers:
                        self.providers[provider_id].update({
                            "display_name": provider_data.get("display_name", provider_id)
                        })
            
            # Load models
            if "models" in config:
                self.models = config["models"]
            
            # Load QA criteria
            if "qa_criteria" in config:
                self.qa_criteria = config["qa_criteria"]
                
            # Load evaluation criteria
            if "eval_criteria" in config:
                self.eval_criteria = config["eval_criteria"]
            
            logger.debug(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.models = {}  # Ensure models is empty on error
            self.qa_criteria = []  # Ensure qa_criteria is empty on error
            self.eval_criteria = []  # Ensure eval_criteria is empty on error
    
    def _get_sorted_model_items(self):
        """Get model items sorted by display name.
        
        Returns:
            List of tuples containing (model_id, model_data)
        """
        return sorted(
            self.models.items(),
            key=lambda x: x[1]["provider"].lower()
        )

    def get_enabled_models(self) -> List[Dict[str, Any]]:
        """Get a list of all enabled models, sorted by display name.
        
        Returns:
            A list of enabled models with their details
        """
        enabled_models = []
        for model_id, model_data in self._get_sorted_model_items():
            if model_data.get("enabled", True):
                model_info = model_data.copy()
                model_info["id"] = model_id
                enabled_models.append(model_info)
        
        return enabled_models
    
    def get_model_dropdown_values(self) -> List[str]:
        """Get a list of display names for all enabled models.
        
        Returns:
            A list of display names for enabled models
        """
        enabled_models = self.get_enabled_models()
        return [model["display_name"] for model in enabled_models]
    
    def get_model_by_display_name(self, display_name: str) -> Optional[Dict[str, Any]]:
        """Get model details by its display name.
        
        Args:
            display_name: The display name of the model
            
        Returns:
            The model details or None if not found
        """
        enabled_models = self.get_enabled_models()
        for model in enabled_models:
            if model["display_name"] == display_name:
                return model
        return None
    
    def set_model_enabled(self, model_id: str, enabled: bool, cache_only: bool = False):
        """Set model enabled state with optional caching."""
        if cache_only:
            self._cached_changes[model_id] = enabled
            self._has_pending_changes = True
            logger.debug("Cached model %s state change to: %s", 
                        model_id, "enabled" if enabled else "disabled")
        else:
            # Direct change as before
            self.models[model_id]["enabled"] = enabled
            self.save_config()
    
    def delete_model(self, model_id: str):
        """Delete a model from the registry.
        
        Args:
            model_id: The ID of the model to delete
        """
        if model_id in self.models:
            del self.models[model_id]
            self.save_config()
            logger.debug("Deleted model: %s", model_id)
    
    def commit_cached_changes(self):
        """Apply all cached changes and save configuration."""
        if not self._has_pending_changes:
            return False
            
        logger.debug("Committing cached changes")
        try:
            # Apply enable/disable changes
            for model_id, enabled in self._cached_changes.items():
                if model_id in self.models:
                    self.models[model_id]["enabled"] = enabled
            
            self.save_config()
            self._cached_changes.clear()
            self._has_pending_changes = False
            return True
        except Exception as e:
            logger.error("Error committing changes: %s", str(e))
            self.discard_cached_changes()
            return False
        
    def discard_cached_changes(self):
        """Discard all cached changes without applying them."""
        if self._has_pending_changes:
            logger.debug("Discarding cached changes")
            self._cached_changes.clear()
            self._has_pending_changes = False
    
    def get_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get all providers.
        
        Returns:
            A dictionary of providers with their details
        """
        return self.providers
    
    def get_models_by_provider(self, provider_id: str) -> List[Dict[str, Any]]:
        """Get all models for a specific provider, sorted by display name.
        
        Args:
            provider_id: The ID of the provider
            
        Returns:
            A list of models for the specified provider
        """
        result = []
        for model_id, model_data in self._get_sorted_model_items():
            if model_data["provider"] == provider_id:
                model_info = model_data.copy()
                model_info["id"] = model_id
                result.append(model_info)
        return result
    
    def get_qa_criteria(self) -> List[Dict[str, Any]]:
        """Get all QA criteria from config.
        
        Returns:
            List of criteria dictionaries
        """
        return self.qa_criteria
    
    def save_qa_criteria(self, criteria: List[Dict[str, Any]]):
        """Save QA criteria to config.
        
        Args:
            criteria: List of criteria dictionaries
        """
        self.qa_criteria = criteria
        self.save_config()
        logger.debug("Saved %d QA criteria to config", len(criteria))
        
    def get_eval_criteria(self) -> List[Dict[str, Any]]:
        """Get all evaluation criteria from config.
        
        Returns:
            List of criteria dictionaries
        """
        return self.eval_criteria
    
    def save_eval_criteria(self, criteria: List[Dict[str, Any]]):
        """Save evaluation criteria to config.
        
        Args:
            criteria: List of criteria dictionaries
        """
        self.eval_criteria = criteria
        self.save_config()
        logger.debug("Saved %d evaluation criteria to config", len(criteria)) 