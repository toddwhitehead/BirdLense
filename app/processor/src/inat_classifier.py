"""
iNaturalist-based bird classifier using the Birder project models.

This classifier supports 10,000+ species worldwide including Australian birds
(cockatoos, king parrots, lorikeets, etc.) that are not in the NABirds dataset.

Models available:
- rope_vit_reg4_b14_capi-inat21: Full model (336x336 input, 90% accuracy)
- rope_vit_reg4_b14_capi-inat21-224px: Smaller variant (224x224 input, 88.6% accuracy)

Installation: pip install birder
"""

import logging
from typing import Optional, Tuple, List, Dict
import numpy as np

logger = logging.getLogger(__name__)

# Try to import birder, provide helpful error if not installed
try:
    import birder
    from birder.inference.classification import infer_image
    BIRDER_AVAILABLE = True
except ImportError:
    BIRDER_AVAILABLE = False
    logger.warning(
        "Birder library not installed. Install with: pip install birder\n"
        "This is required for global bird species classification (including Australian birds)."
    )


class INatClassifier:
    """
    Bird classifier using iNaturalist 2021 trained models from the Birder project.
    
    Supports 10,000 species including 1,486 bird species worldwide.
    Australian species included: Sulphur-crested Cockatoo, Australian King-Parrot, 
    Rainbow Lorikeet, Galah, Kookaburra, etc.
    """
    
    # Common name mappings for Australian birds (scientific -> common name)
    # The iNaturalist model uses scientific names, we map to common names for display
    COMMON_NAMES: Dict[str, str] = {
        # Cockatoos
        "Cacatua galerita": "Sulphur-crested Cockatoo",
        "Cacatua sanguinea": "Little Corella",
        "Cacatua tenuirostris": "Long-billed Corella",
        "Eolophus roseicapilla": "Galah",
        "Calyptorhynchus banksii": "Red-tailed Black Cockatoo",
        "Zanda funerea": "Yellow-tailed Black Cockatoo",
        "Nymphicus hollandicus": "Cockatiel",
        # Parrots
        "Alisterus scapularis": "Australian King-Parrot",
        "Platycercus elegans": "Crimson Rosella",
        "Platycercus eximius": "Eastern Rosella",
        "Trichoglossus moluccanus": "Rainbow Lorikeet",
        "Trichoglossus chlorolepidotus": "Scaly-breasted Lorikeet",
        "Melopsittacus undulatus": "Budgerigar",
        "Glossopsitta concinna": "Musk Lorikeet",
        # Other Australian birds
        "Dacelo novaeguineae": "Laughing Kookaburra",
        "Gymnorhina tibicen": "Australian Magpie",
        "Cracticus torquatus": "Grey Butcherbird",
        "Corvus coronoides": "Australian Raven",
        "Manorina melanocephala": "Noisy Miner",
        "Acridotheres tristis": "Common Myna",
        "Sturnus vulgaris": "Common Starling",
        "Passer domesticus": "House Sparrow",
    }
    
    def __init__(
        self, 
        model_name: str = "rope_vit_reg4_b14_capi-inat21-224px",
        bird_only: bool = True,
        regional_species: Optional[List[str]] = None
    ):
        """
        Initialize the iNaturalist classifier.
        
        Args:
            model_name: Birder model name. Options:
                - "rope_vit_reg4_b14_capi-inat21" (336x336, more accurate)
                - "rope_vit_reg4_b14_capi-inat21-224px" (224x224, faster)
            bird_only: If True, only return bird classifications (class "Aves")
            regional_species: Optional list of species/families to filter for
        """
        self.model_name = model_name
        self.bird_only = bird_only
        self.regional_species = regional_species
        self.net = None
        self.model_info = None
        self.transform = None
        self.size = None
        
        if not BIRDER_AVAILABLE:
            raise ImportError(
                "Birder library not installed. Install with: pip install birder"
            )
        
        self._load_model()
    
    def _load_model(self):
        """Load the Birder model and create transform."""
        logger.info(f"Loading iNaturalist classifier: {self.model_name}")
        
        try:
            self.net, self.model_info = birder.load_pretrained_model(
                self.model_name, 
                inference=True
            )
            self.size = birder.get_size_from_signature(self.model_info.signature)
            self.transform = birder.classification_transform(
                self.size, 
                self.model_info.rgb_stats
            )
            logger.info(
                f"iNaturalist classifier loaded: {self.model_name} "
                f"(input size: {self.size}x{self.size})"
            )
        except Exception as e:
            logger.error(f"Failed to load iNaturalist model: {e}")
            raise
    
    def _get_common_name(self, scientific_name: str) -> str:
        """
        Convert scientific name to common name if available.
        
        Args:
            scientific_name: Scientific name (e.g., "Cacatua galerita")
            
        Returns:
            Common name if available, otherwise cleaned scientific name
        """
        # Check our mapping first
        if scientific_name in self.COMMON_NAMES:
            return self.COMMON_NAMES[scientific_name]
        
        # Otherwise return the scientific name, cleaned up
        # iNat format: "Aves_Cacatua_galerita" -> "Cacatua galerita"
        parts = scientific_name.split("_")
        if len(parts) >= 2:
            # Skip taxonomy prefix if present (Aves_, Mammalia_, etc.)
            if parts[0] in ["Aves", "Mammalia", "Reptilia", "Amphibia", "Insecta"]:
                parts = parts[1:]
            return " ".join(parts)
        
        return scientific_name.replace("_", " ")
    
    def _is_bird_class(self, class_name: str) -> bool:
        """Check if a class belongs to birds (Aves)."""
        # iNaturalist class names include taxonomy: "Aves/..." or start with bird families
        return class_name.startswith("Aves") or "_Aves_" in class_name
    
    def classify(self, crop: np.ndarray) -> Tuple[Optional[str], float, dict]:
        """
        Classify a bird crop using the iNaturalist model.
        
        Args:
            crop: BGR image crop of the detected bird
            
        Returns:
            Tuple of (species_name, confidence, metadata)
            - species_name: Common name of the species (or None if not a bird)
            - confidence: Confidence score (0.0 to 1.0)
            - metadata: Dict with additional info (scientific_name, top_predictions)
        """
        if crop is None or crop.size == 0:
            return None, 0.0, {}
        
        try:
            # Convert BGR to RGB (birder expects RGB)
            from PIL import Image
            rgb_crop = crop[:, :, ::-1]  # BGR to RGB
            pil_image = Image.fromarray(rgb_crop)
            
            # Run inference
            out, embedding = infer_image(
                self.net, 
                pil_image, 
                self.transform,
                return_embedding=True
            )
            
            # out shape: (1, 10000) - probabilities for each class
            probs = out[0]
            
            # Get top predictions
            top_k = 5
            top_indices = np.argsort(probs)[::-1][:top_k]
            
            # Get class names from model
            class_names = self.model_info.class_to_idx
            idx_to_class = {v: k for k, v in class_names.items()}
            
            top_predictions = []
            for idx in top_indices:
                if idx in idx_to_class:
                    scientific_name = idx_to_class[idx]
                    common_name = self._get_common_name(scientific_name)
                    conf = float(probs[idx])
                    top_predictions.append({
                        "scientific_name": scientific_name,
                        "common_name": common_name,
                        "confidence": conf
                    })
            
            # Filter for birds if bird_only is enabled
            if self.bird_only:
                bird_predictions = [
                    p for p in top_predictions 
                    if self._is_bird_class(p["scientific_name"])
                ]
                if bird_predictions:
                    top_predictions = bird_predictions
            
            # Filter for regional species if configured
            if self.regional_species:
                filtered = []
                for pred in top_predictions:
                    name_lower = pred["common_name"].lower()
                    sci_lower = pred["scientific_name"].lower()
                    for rs in self.regional_species:
                        rs_lower = rs.lower()
                        if rs_lower in name_lower or rs_lower in sci_lower:
                            filtered.append(pred)
                            break
                if filtered:
                    top_predictions = filtered
            
            if not top_predictions:
                return None, 0.0, {"top_predictions": []}
            
            best = top_predictions[0]
            
            # Log results
            top3_str = ", ".join([
                f"{p['common_name']}:{p['confidence']:.1%}" 
                for p in top_predictions[:3]
            ])
            logger.info(
                f"iNat classification: {best['common_name']} ({best['confidence']:.1%}) | "
                f"Top 3: [{top3_str}]"
            )
            
            return (
                best["common_name"],
                best["confidence"],
                {
                    "scientific_name": best["scientific_name"],
                    "top_predictions": top_predictions
                }
            )
            
        except Exception as e:
            logger.error(f"iNaturalist classification failed: {e}", exc_info=True)
            return None, 0.0, {"error": str(e)}


def create_inat_classifier(
    model_name: str = "rope_vit_reg4_b14_capi-inat21-224px",
    bird_only: bool = True,
    regional_species: Optional[List[str]] = None
) -> Optional[INatClassifier]:
    """
    Factory function to create an iNaturalist classifier.
    
    Returns None if birder is not available.
    """
    if not BIRDER_AVAILABLE:
        logger.error(
            "Cannot create iNaturalist classifier: birder library not installed.\n"
            "Install with: pip install birder"
        )
        return None
    
    try:
        return INatClassifier(
            model_name=model_name,
            bird_only=bird_only,
            regional_species=regional_species
        )
    except Exception as e:
        logger.error(f"Failed to create iNaturalist classifier: {e}")
        return None
