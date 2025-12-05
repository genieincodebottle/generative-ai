"""
ML-based Content Classification using HateBERT and other transformer models.

This module provides ML-powered toxicity and hate speech detection,
replacing keyword-based detection with semantic understanding.

Configuration via environment variables:
- USE_ML_MODELS: Enable/disable ML models (true/false)
- ML_PRIMARY_MODEL: Primary model to use
- ML_USE_ENSEMBLE: Use multiple models (true/false)
- ML_PRELOAD_MODELS: Load models at startup (true/false)
- ML_DEVICE: Device for inference (auto/cpu/cuda/mps)
- ML_MODELS_CACHE_DIR: Directory for model cache
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_str(key: str, default: str = "") -> str:
    """Get string value from environment variable."""
    return os.getenv(key, default)


# Configuration from environment
class MLConfig:
    """ML Configuration from environment variables."""

    @staticmethod
    def is_ml_enabled() -> bool:
        """Check if ML models are enabled."""
        return get_env_bool("USE_ML_MODELS", False)

    @staticmethod
    def get_primary_model() -> str:
        """Get primary model name."""
        return get_env_str("ML_PRIMARY_MODEL", "distilbert_toxic")

    @staticmethod
    def use_ensemble() -> bool:
        """Check if ensemble mode is enabled."""
        return get_env_bool("ML_USE_ENSEMBLE", False)

    @staticmethod
    def should_preload() -> bool:
        """Check if models should be preloaded at startup."""
        return get_env_bool("ML_PRELOAD_MODELS", True)

    @staticmethod
    def get_device() -> str:
        """Get device for inference."""
        return get_env_str("ML_DEVICE", "auto")

    @staticmethod
    def get_cache_dir() -> str:
        """Get model cache directory."""
        return get_env_str("ML_MODELS_CACHE_DIR", "./models_cache")


# Check if ML is enabled via config
ML_ENABLED = MLConfig.is_ml_enabled()

# Try to import ML libraries only if enabled
ML_AVAILABLE = False
if ML_ENABLED:
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline
        )
        ML_AVAILABLE = True
        logger.info("ML libraries loaded successfully (USE_ML_MODELS=true)")
    except ImportError as e:
        logger.warning(f"ML libraries not available: {e}. Install with: pip install transformers torch")
        ML_AVAILABLE = False
else:
    logger.info("ML models disabled (USE_ML_MODELS=false). Using keyword-based detection.")


class ToxicityCategory(str, Enum):
    """Categories of toxic content detected by ML models."""
    TOXIC = "toxic"
    SEVERE_TOXIC = "severe_toxic"
    OBSCENE = "obscene"
    THREAT = "threat"
    INSULT = "insult"
    IDENTITY_HATE = "identity_hate"
    HATE_SPEECH = "hate_speech"
    OFFENSIVE = "offensive"
    SAFE = "safe"


@dataclass
class MLPrediction:
    """Structured prediction result from ML classifier."""
    label: str
    score: float
    category: ToxicityCategory
    raw_scores: Dict[str, float]


class ContentMLClassifier:
    """
    ML-based content classifier using HateBERT and other transformer models.

    Models used:
    1. HateBERT (GroNLP/hateBERT) - Fine-tuned BERT for hate speech detection
    2. Toxic-BERT (unitary/toxic-bert) - Multi-label toxicity classification
    3. RoBERTa Hate Speech (facebook/roberta-hate-speech-dynabench-r4-target) - Hate speech detection

    Falls back to keyword-based detection if ML libraries are not available.
    """

    # Model configurations
    MODELS = {
        "hatebert": {
            "name": "GroNLP/hateBERT",
            "task": "text-classification",
            "description": "HateBERT for hate speech detection"
        },
        "toxic_bert": {
            "name": "unitary/toxic-bert",
            "task": "text-classification",
            "description": "Multi-label toxicity classification"
        },
        "roberta_hate": {
            "name": "facebook/roberta-hate-speech-dynabench-r4-target",
            "task": "text-classification",
            "description": "RoBERTa for hate speech"
        },
        "distilbert_toxic": {
            "name": "martin-ha/toxic-comment-model",
            "task": "text-classification",
            "description": "DistilBERT toxic comment classifier"
        }
    }

    def __init__(
        self,
        primary_model: Optional[str] = None,
        use_ensemble: Optional[bool] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        auto_load: bool = True
    ):
        """
        Initialize the ML classifier.

        Args:
            primary_model: Which model to use (default from env: ML_PRIMARY_MODEL)
            use_ensemble: Whether to use multiple models (default from env: ML_USE_ENSEMBLE)
            device: Device to run models on (default from env: ML_DEVICE)
            cache_dir: Directory to cache models (default from env: ML_MODELS_CACHE_DIR)
            auto_load: Whether to automatically load models (set False for manual preload)
        """
        # Use environment config as defaults
        self.primary_model = primary_model or MLConfig.get_primary_model()
        self.use_ensemble = use_ensemble if use_ensemble is not None else MLConfig.use_ensemble()
        self.models_loaded = False
        self.classifiers: Dict[str, Any] = {}
        self.ml_enabled = MLConfig.is_ml_enabled()

        # Determine device
        env_device = MLConfig.get_device()
        if device:
            self.device = device
        elif env_device and env_device != "auto":
            self.device = env_device
        elif ML_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
        elif ML_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.cache_dir = cache_dir or MLConfig.get_cache_dir() or os.path.join(
            os.path.dirname(__file__), "..", "models_cache"
        )

        # Log configuration
        logger.info(f"ML Classifier Config:")
        logger.info(f"  - ML Enabled: {self.ml_enabled}")
        logger.info(f"  - Primary Model: {self.primary_model}")
        logger.info(f"  - Ensemble Mode: {self.use_ensemble}")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Cache Dir: {self.cache_dir}")

        # Load models if ML is enabled and available
        if auto_load and self.ml_enabled and ML_AVAILABLE:
            self._load_models()
        elif not self.ml_enabled:
            logger.info("ML models not loaded (disabled via USE_ML_MODELS=false)")

    def _load_models(self) -> None:
        """Load the transformer models."""
        try:
            logger.info(f"Loading ML models on device: {self.device}")

            # Load primary model
            if self.primary_model in self.MODELS:
                model_config = self.MODELS[self.primary_model]
                logger.info(f"Loading primary model: {model_config['name']}")

                self.classifiers[self.primary_model] = pipeline(
                    model_config["task"],
                    model=model_config["name"],
                    device=0 if self.device == "cuda" else -1,
                    truncation=True,
                    max_length=512
                )
                logger.info(f"Loaded {self.primary_model} successfully")

            # Load ensemble models if requested
            if self.use_ensemble:
                for model_key, model_config in self.MODELS.items():
                    if model_key != self.primary_model and model_key not in self.classifiers:
                        try:
                            logger.info(f"Loading ensemble model: {model_config['name']}")
                            self.classifiers[model_key] = pipeline(
                                model_config["task"],
                                model=model_config["name"],
                                device=0 if self.device == "cuda" else -1,
                                truncation=True,
                                max_length=512
                            )
                        except Exception as e:
                            logger.warning(f"Failed to load {model_key}: {e}")

            self.models_loaded = len(self.classifiers) > 0
            logger.info(f"ML classifier ready with {len(self.classifiers)} model(s)")

        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            self.models_loaded = False

    def predict_toxicity(self, text: str) -> Dict[str, Any]:
        """
        Predict toxicity of text using ML models.

        Args:
            text: The text content to analyze

        Returns:
            Dictionary with toxicity analysis results
        """
        if not self.models_loaded:
            logger.warning("ML models not loaded, using fallback")
            return self._fallback_prediction(text)

        try:
            results = {}
            all_scores = []

            # Get predictions from each loaded model
            for model_key, classifier in self.classifiers.items():
                try:
                    prediction = classifier(text)

                    # Handle different output formats
                    if isinstance(prediction, list) and len(prediction) > 0:
                        pred = prediction[0]
                        label = pred.get('label', 'unknown')
                        score = pred.get('score', 0.0)

                        # Log raw prediction for debugging
                        logger.info(f"ML [{model_key}] raw: label='{label}', score={score:.4f}, text='{text[:50]}...'")

                        # Normalize labels across different models
                        normalized = self._normalize_prediction(label, score, model_key)
                        results[model_key] = normalized
                        all_scores.append(normalized['toxicity_score'])

                        logger.info(f"ML [{model_key}] normalized: toxicity_score={normalized['toxicity_score']:.4f}")

                except Exception as e:
                    logger.warning(f"Error with model {model_key}: {e}")

            if not results:
                return self._fallback_prediction(text)

            # Combine results
            combined = self._combine_predictions(results, all_scores)
            return combined

        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._fallback_prediction(text)

    def _normalize_prediction(
        self,
        label: str,
        score: float,
        model_key: str
    ) -> Dict[str, Any]:
        """
        Normalize prediction output across different models.

        Different models use different label conventions:
        - toxic-bert: "toxic", "non-toxic"
        - hateBERT: "HATE", "NOT HATE"
        - roberta: "hate", "nothate"
        - martin-ha/toxic-comment-model: "toxic", "non-toxic"
        """
        label_lower = label.lower()

        # Check for explicit "non-toxic" or "not" patterns first (these indicate SAFE content)
        is_safe = any(x in label_lower for x in ['non-toxic', 'non_toxic', 'nontoxic', 'not_hate', 'not hate', 'nothate', 'safe', 'neutral', 'label_0'])

        # Check for toxic patterns
        is_toxic = any(x in label_lower for x in ['toxic', 'hate', 'offensive', 'label_1']) and not is_safe

        # Log for debugging
        logger.debug(f"Label normalization: label='{label}', score={score}, is_toxic={is_toxic}, is_safe={is_safe}")

        if is_toxic:
            # Model says it's toxic with this confidence
            toxicity_score = score
        elif is_safe:
            # Model says it's safe with this confidence, so toxicity is the inverse
            toxicity_score = 1.0 - score
        else:
            # Unknown label - use score directly if > 0.5, else assume safe
            toxicity_score = score if score > 0.5 else 1.0 - score

        # Determine category
        if 'hate' in label_lower:
            category = ToxicityCategory.HATE_SPEECH
        elif 'severe' in label_lower:
            category = ToxicityCategory.SEVERE_TOXIC
        elif 'threat' in label_lower:
            category = ToxicityCategory.THREAT
        elif 'insult' in label_lower:
            category = ToxicityCategory.INSULT
        elif 'obscene' in label_lower:
            category = ToxicityCategory.OBSCENE
        elif toxicity_score > 0.5:
            category = ToxicityCategory.TOXIC
        else:
            category = ToxicityCategory.SAFE

        return {
            "original_label": label,
            "original_score": score,
            "toxicity_score": toxicity_score,
            "category": category.value,
            "model": model_key
        }

    def _combine_predictions(
        self,
        results: Dict[str, Dict[str, Any]],
        scores: List[float]
    ) -> Dict[str, Any]:
        """Combine predictions from multiple models."""

        # Calculate ensemble score (weighted average)
        weights = {
            "distilbert_toxic": 1.0,
            "hatebert": 0.9,
            "toxic_bert": 0.85,
            "roberta_hate": 0.8
        }

        weighted_sum = 0.0
        weight_total = 0.0

        for model_key, result in results.items():
            weight = weights.get(model_key, 0.5)
            weighted_sum += result['toxicity_score'] * weight
            weight_total += weight

        ensemble_score = weighted_sum / weight_total if weight_total > 0 else 0.0

        # Determine categories from all models
        categories = [r['category'] for r in results.values() if r['category'] != 'safe']

        # Determine toxicity level
        if ensemble_score >= 0.9:
            level = "severe"
        elif ensemble_score >= 0.7:
            level = "high"
        elif ensemble_score >= 0.5:
            level = "medium"
        elif ensemble_score >= 0.3:
            level = "low"
        else:
            level = "none"

        return {
            "toxicity_score": round(ensemble_score, 4),
            "toxicity_level": level,
            "categories": list(set(categories)) if categories else ["safe"],
            "is_toxic": ensemble_score >= 0.5,
            "is_severe": ensemble_score >= 0.8,
            "confidence": self._calculate_confidence(results, scores),
            "model_predictions": results,
            "models_used": list(results.keys()),
            "detection_method": "ml_ensemble" if len(results) > 1 else "ml_single"
        }

    def _calculate_confidence(
        self,
        results: Dict[str, Dict[str, Any]],
        scores: List[float]
    ) -> float:
        """Calculate confidence based on model agreement."""
        if len(scores) <= 1:
            return 0.85  # Single model confidence

        # Calculate variance in predictions
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        # Higher variance = lower confidence
        # Max variance is 0.25 (when scores are 0 and 1)
        confidence = 1.0 - (variance / 0.25) * 0.3  # 30% confidence reduction max

        return round(max(0.5, min(1.0, confidence)), 2)

    def _fallback_prediction(self, text: str) -> Dict[str, Any]:
        """
        Fallback keyword-based prediction when ML is not available.
        This preserves the original keyword-based logic as a backup.
        """
        from ..ml.keyword_detectors import keyword_toxicity_detection, keyword_hate_speech_detection

        toxicity_result = keyword_toxicity_detection(text)
        hate_result = keyword_hate_speech_detection(text)

        # Combine keyword-based scores
        combined_score = max(
            toxicity_result['toxicity_score'],
            hate_result['score']
        )

        categories = toxicity_result['categories'].copy()
        if hate_result['detected']:
            categories.append('hate_speech')

        # Determine level
        if combined_score >= 0.9:
            level = "severe"
        elif combined_score >= 0.7:
            level = "high"
        elif combined_score >= 0.5:
            level = "medium"
        elif combined_score >= 0.3:
            level = "low"
        else:
            level = "none"

        return {
            "toxicity_score": round(combined_score, 4),
            "toxicity_level": level,
            "categories": categories if categories else ["safe"],
            "is_toxic": combined_score >= 0.5,
            "is_severe": combined_score >= 0.8,
            "confidence": 0.7,  # Lower confidence for keyword-based
            "model_predictions": {
                "keyword_toxicity": toxicity_result,
                "keyword_hate": hate_result
            },
            "models_used": ["keyword_fallback"],
            "detection_method": "keyword_fallback",
            # Include raw keyword counts for debugging
            "profanity_count": toxicity_result.get('profanity_count', 0),
            "insult_count": toxicity_result.get('insult_count', 0),
            "threat_count": toxicity_result.get('threat_count', 0)
        }

    def predict_hate_speech(self, text: str) -> Dict[str, Any]:
        """
        Specialized hate speech detection using HateBERT.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with hate speech analysis
        """
        if not self.models_loaded:
            return self._fallback_hate_speech(text)

        try:
            # Use HateBERT specifically if available
            if 'hatebert' in self.classifiers:
                result = self.classifiers['hatebert'](text)
                if result and len(result) > 0:
                    pred = result[0]
                    is_hate = 'hate' in pred['label'].lower()
                    score = pred['score'] if is_hate else 1.0 - pred['score']

                    return {
                        "is_hate_speech": score >= 0.5,
                        "hate_score": round(score, 4),
                        "confidence": round(pred['score'], 4),
                        "raw_label": pred['label'],
                        "detection_method": "hatebert"
                    }

            # Fall back to general toxicity
            general = self.predict_toxicity(text)
            return {
                "is_hate_speech": 'hate_speech' in general.get('categories', []),
                "hate_score": general['toxicity_score'] if 'hate_speech' in general.get('categories', []) else 0.0,
                "confidence": general['confidence'],
                "detection_method": "general_toxicity"
            }

        except Exception as e:
            logger.error(f"Hate speech prediction failed: {e}")
            return self._fallback_hate_speech(text)

    def _fallback_hate_speech(self, text: str) -> Dict[str, Any]:
        """Fallback hate speech detection using keywords."""
        from ..ml.keyword_detectors import keyword_hate_speech_detection
        result = keyword_hate_speech_detection(text)

        return {
            "is_hate_speech": result['detected'],
            "hate_score": round(result['score'], 4),
            "confidence": 0.6,
            "patterns": result['patterns'],
            "detection_method": "keyword_fallback"
        }

    def analyze_content(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive content analysis combining all ML predictions.

        Args:
            text: Text content to analyze

        Returns:
            Complete analysis with toxicity, hate speech, and recommendations
        """
        # Get toxicity prediction
        toxicity = self.predict_toxicity(text)

        # Get specialized hate speech prediction
        hate_speech = self.predict_hate_speech(text)

        # Combine into comprehensive analysis
        combined_score = max(toxicity['toxicity_score'], hate_speech['hate_score'])

        # Determine recommended action
        if combined_score >= 0.85:
            recommended_action = "remove"
            action_confidence = 0.9
        elif combined_score >= 0.7:
            recommended_action = "flag_for_review"
            action_confidence = 0.8
        elif combined_score >= 0.5:
            recommended_action = "warn"
            action_confidence = 0.75
        else:
            recommended_action = "approve"
            action_confidence = 0.85

        return {
            "toxicity_analysis": toxicity,
            "hate_speech_analysis": hate_speech,
            "combined_score": round(combined_score, 4),
            "recommended_action": recommended_action,
            "action_confidence": action_confidence,
            "requires_human_review": combined_score >= 0.6 and combined_score < 0.85,
            "auto_remove": combined_score >= 0.9,
            "ml_available": self.models_loaded
        }


# Singleton instance for reuse
_classifier_instance: Optional[ContentMLClassifier] = None
_preload_complete: bool = False


def get_ml_classifier(
    primary_model: Optional[str] = None,
    use_ensemble: Optional[bool] = None
) -> Optional[ContentMLClassifier]:
    """
    Get or create the ML classifier singleton.

    Args:
        primary_model: Primary model to use (default from env)
        use_ensemble: Whether to use ensemble of models (default from env)

    Returns:
        ContentMLClassifier instance or None if ML is disabled
    """
    global _classifier_instance

    # Check if ML is enabled
    if not MLConfig.is_ml_enabled():
        logger.debug("ML classifier requested but USE_ML_MODELS=false")
        return None

    if _classifier_instance is None:
        _classifier_instance = ContentMLClassifier(
            primary_model=primary_model,
            use_ensemble=use_ensemble
        )

    return _classifier_instance


def preload_ml_models() -> Dict[str, Any]:
    """
    Preload ML models at application startup.

    Call this function during app initialization to load models
    before handling any requests. This prevents cold-start latency.

    Returns:
        Dict with preload status information
    """
    global _classifier_instance, _preload_complete

    result = {
        "ml_enabled": MLConfig.is_ml_enabled(),
        "preload_requested": MLConfig.should_preload(),
        "models_loaded": False,
        "primary_model": MLConfig.get_primary_model(),
        "device": "unknown",
        "error": None
    }

    if not MLConfig.is_ml_enabled():
        logger.info("ML preload skipped (USE_ML_MODELS=false)")
        result["status"] = "skipped"
        return result

    if not MLConfig.should_preload():
        logger.info("ML preload skipped (ML_PRELOAD_MODELS=false)")
        result["status"] = "skipped"
        return result

    if not ML_AVAILABLE:
        logger.warning("ML preload failed: transformers/torch not installed")
        result["status"] = "failed"
        result["error"] = "ML libraries not installed"
        return result

    try:
        logger.info("Preloading ML models...")

        # Create classifier (this loads the models)
        _classifier_instance = ContentMLClassifier(
            primary_model=MLConfig.get_primary_model(),
            use_ensemble=MLConfig.use_ensemble(),
            auto_load=True
        )

        result["models_loaded"] = _classifier_instance.models_loaded
        result["device"] = _classifier_instance.device
        result["models_available"] = list(_classifier_instance.classifiers.keys())

        if _classifier_instance.models_loaded:
            _preload_complete = True
            logger.info(f"ML models preloaded successfully on {_classifier_instance.device}")
            result["status"] = "success"
        else:
            logger.warning("ML preload completed but no models loaded")
            result["status"] = "partial"

    except Exception as e:
        logger.error(f"ML preload failed: {e}")
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def is_ml_ready() -> bool:
    """Check if ML models are loaded and ready for inference."""
    if not MLConfig.is_ml_enabled():
        return False
    if _classifier_instance is None:
        return False
    return _classifier_instance.models_loaded


def get_ml_status() -> Dict[str, Any]:
    """Get current ML classifier status."""
    return {
        "ml_enabled": MLConfig.is_ml_enabled(),
        "ml_available": ML_AVAILABLE,
        "models_loaded": _classifier_instance.models_loaded if _classifier_instance else False,
        "preload_complete": _preload_complete,
        "primary_model": MLConfig.get_primary_model(),
        "ensemble_mode": MLConfig.use_ensemble(),
        "device": _classifier_instance.device if _classifier_instance else "none",
        "loaded_models": list(_classifier_instance.classifiers.keys()) if _classifier_instance else []
    }


# Convenience functions for direct use
def ml_detect_toxicity(text: str) -> Dict[str, Any]:
    """Detect toxicity using ML classifier."""
    classifier = get_ml_classifier()
    if classifier:
        return classifier.predict_toxicity(text)
    # Return fallback result
    return _fallback_toxicity(text)


def ml_detect_hate_speech(text: str) -> Dict[str, Any]:
    """Detect hate speech using ML classifier."""
    classifier = get_ml_classifier()
    if classifier:
        return classifier.predict_hate_speech(text)
    # Return fallback result
    return _fallback_hate_speech(text)


def ml_analyze_content(text: str) -> Dict[str, Any]:
    """Complete content analysis using ML classifier."""
    classifier = get_ml_classifier()
    if classifier:
        return classifier.analyze_content(text)
    # Return basic analysis without ML
    toxicity = _fallback_toxicity(text)
    hate = _fallback_hate_speech(text)
    return {
        "toxicity_analysis": toxicity,
        "hate_speech_analysis": hate,
        "combined_score": max(toxicity.get("toxicity_score", 0), hate.get("score", 0)),
        "ml_available": False
    }


def _fallback_toxicity(text: str) -> Dict[str, Any]:
    """Fallback toxicity detection when ML is disabled."""
    from ..ml.keyword_detectors import keyword_toxicity_detection
    return keyword_toxicity_detection(text)


def _fallback_hate_speech(text: str) -> Dict[str, Any]:
    """Fallback hate speech detection when ML is disabled."""
    from ..ml.keyword_detectors import keyword_hate_speech_detection
    return keyword_hate_speech_detection(text)
