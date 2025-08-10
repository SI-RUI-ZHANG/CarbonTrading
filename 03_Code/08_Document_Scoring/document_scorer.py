"""Document scorer with spectrum positioning algorithm."""

from typing import Dict, Optional, Tuple
import numpy as np
import config

class DocumentScorer:
    """Score documents using spectrum positioning based on anchor similarities."""
    
    def __init__(self):
        self.spectrum_positions = config.SPECTRUM_POSITIONS
    
    def score_document(self, similarities: Dict) -> Dict:
        """
        Calculate spectrum scores for a document based on similarities to anchors.
        
        Args:
            similarities: Dictionary with similarity scores for all dimensions and categories
            
        Returns:
            Dictionary with spectrum scores for each dimension
        """
        if not similarities:
            return self._get_default_scores()
        
        scores = {}
        
        # Calculate spectrum position for each dimension
        for dimension in ['supply', 'demand', 'policy_strength']:
            if dimension in similarities:
                score = self.calculate_spectrum_score(similarities[dimension], dimension)
                scores[dimension] = score
            else:
                scores[dimension] = 0
        
        # Add confidence if available
        if 'confidence' in similarities:
            scores['confidence'] = similarities['confidence']
        else:
            scores['confidence'] = self._calculate_confidence(similarities)
        
        # Add interpretations
        scores['interpretations'] = self._interpret_scores(scores)
        
        return scores
    
    def calculate_spectrum_score(self, category_similarities: Dict, dimension: str) -> float:
        """
        Calculate position on spectrum using weighted interpolation.
        
        Args:
            category_similarities: Similarity scores for each category
            dimension: The dimension being scored
            
        Returns:
            Position on spectrum (-100 to 100 for supply/demand, 0 to 100 for policy_strength)
        """
        # Get total similarity
        total_similarity = sum(category_similarities.values())
        
        # If no similarity to any anchor, return neutral position
        if total_similarity == 0:
            return 0
        
        # Calculate normalized weights
        weights = {
            cat: sim / total_similarity 
            for cat, sim in category_similarities.items()
        }
        
        # Get spectrum positions for this dimension
        positions = self.spectrum_positions[dimension]
        
        # Calculate weighted position
        spectrum_score = sum(
            weights.get(cat, 0) * positions.get(cat, 0) 
            for cat in positions
        )
        
        return round(spectrum_score, 2)
    
    def _calculate_confidence(self, similarities: Dict) -> float:
        """
        Calculate confidence based on similarity strength and consistency.
        
        Args:
            similarities: All similarity scores
            
        Returns:
            Confidence score (0-100)
        """
        all_scores = []
        for dimension in ['supply', 'demand', 'policy_strength']:
            if dimension in similarities:
                all_scores.extend(similarities[dimension].values())
        
        if not all_scores:
            return 0
        
        # Higher max similarity = higher confidence
        max_similarity = max(all_scores)
        
        # Lower variance = higher confidence (more decisive)
        variance = np.var(all_scores)
        variance_factor = max(0, 1 - variance / 1000)  # Normalize variance impact
        
        # Combine factors
        confidence = (max_similarity * 0.7 + variance_factor * 100 * 0.3)
        
        return round(min(100, max(0, confidence)), 2)
    
    def _interpret_scores(self, scores: Dict) -> Dict:
        """
        Provide human-readable interpretations of spectrum scores.
        
        Args:
            scores: Calculated spectrum scores
            
        Returns:
            Dictionary with interpretations
        """
        interpretations = {}
        
        # Supply interpretation
        supply_score = scores.get('supply', 0)
        if supply_score <= -75:
            interpretations['supply'] = "Major supply decrease"
        elif supply_score <= -25:
            interpretations['supply'] = "Minor supply decrease"
        elif supply_score <= 25:
            interpretations['supply'] = "Neutral supply impact"
        elif supply_score <= 75:
            interpretations['supply'] = "Minor supply increase"
        else:
            interpretations['supply'] = "Major supply increase"
        
        # Demand interpretation
        demand_score = scores.get('demand', 0)
        if demand_score <= -75:
            interpretations['demand'] = "Major demand decrease"
        elif demand_score <= -25:
            interpretations['demand'] = "Minor demand decrease"
        elif demand_score <= 25:
            interpretations['demand'] = "Neutral demand impact"
        elif demand_score <= 75:
            interpretations['demand'] = "Minor demand increase"
        else:
            interpretations['demand'] = "Major demand increase"
        
        # Policy strength interpretation
        policy_score = scores.get('policy_strength', 0)
        if policy_score <= 25:
            interpretations['policy_strength'] = "Informational policy"
        elif policy_score <= 50:
            interpretations['policy_strength'] = "Encouraged/voluntary policy"
        elif policy_score <= 75:
            interpretations['policy_strength'] = "Binding guidance"
        else:
            interpretations['policy_strength'] = "Mandatory enforcement"
        
        # Overall assessment
        interpretations['overall'] = self._get_overall_assessment(scores)
        
        return interpretations
    
    def _get_overall_assessment(self, scores: Dict) -> str:
        """Generate overall assessment based on all scores."""
        supply = scores.get('supply', 0)
        demand = scores.get('demand', 0)
        policy = scores.get('policy_strength', 0)
        confidence = scores.get('confidence', 0)
        
        # Determine price pressure
        # Positive supply = more supply = downward price pressure
        # Positive demand = more demand = upward price pressure
        price_pressure = demand - supply
        
        if abs(price_pressure) < 25:
            price_impact = "neutral price impact"
        elif price_pressure > 75:
            price_impact = "strong upward price pressure"
        elif price_pressure > 25:
            price_impact = "moderate upward price pressure"
        elif price_pressure < -75:
            price_impact = "strong downward price pressure"
        else:
            price_impact = "moderate downward price pressure"
        
        # Combine with policy strength
        if policy > 75:
            enforcement = "with mandatory enforcement"
        elif policy > 50:
            enforcement = "with binding guidance"
        elif policy > 25:
            enforcement = "with voluntary measures"
        else:
            enforcement = "as informational guidance"
        
        # Add confidence qualifier
        if confidence > 80:
            confidence_qual = "High confidence"
        elif confidence > 60:
            confidence_qual = "Moderate confidence"
        else:
            confidence_qual = "Low confidence"
        
        return f"{confidence_qual}: {price_impact} {enforcement}"
    
    def _get_default_scores(self) -> Dict:
        """Return default scores when no similarities are available."""
        return {
            'supply': 0,
            'demand': 0,
            'policy_strength': 0,
            'confidence': 0,
            'interpretations': {
                'supply': "No supply impact detected",
                'demand': "No demand impact detected",
                'policy_strength': "No policy strength detected",
                'overall': "Unable to score document"
            }
        }