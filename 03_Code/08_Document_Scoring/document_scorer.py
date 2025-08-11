"""Document scorer using direct position placement."""

from typing import Dict, Optional
import config

class DocumentScorer:
    """Score documents using direct spectrum positioning from LLM."""
    
    def __init__(self):
        # No longer need spectrum_positions since LLM provides direct positions
        pass
    
    def score_document(self, positions: Dict) -> Dict:
        """
        Process document positions from LLM.
        
        Args:
            positions: Dictionary with direct positions on each spectrum
            
        Returns:
            Dictionary with spectrum scores for each dimension
        """
        if not positions:
            return self._get_default_scores()
        
        # Direct passthrough of positions as scores
        scores = {
            'supply': positions.get('supply', 0),
            'demand': positions.get('demand', 0),
            'policy_strength': positions.get('policy_strength', 0)
        }
        
        # Add interpretations
        scores['interpretations'] = self._interpret_scores(scores)
        
        return scores
    
    def _interpret_scores(self, scores: Dict) -> Dict:
        """
        Provide human-readable interpretations of spectrum scores.
        
        Args:
            scores: Calculated spectrum scores
            
        Returns:
            Dictionary with interpretations
        """
        interpretations = {}
        
        # Supply interpretation (extended range)
        supply_score = scores.get('supply', 0)
        if supply_score <= -110:
            interpretations['supply'] = "Extreme supply restriction"
        elif supply_score <= -75:
            interpretations['supply'] = "Strong supply restriction"
        elif supply_score <= -25:
            interpretations['supply'] = "Supply reduction"
        elif supply_score <= 25:
            interpretations['supply'] = "Neutral supply impact"
        elif supply_score <= 75:
            interpretations['supply'] = "Supply increase"
        elif supply_score <= 110:
            interpretations['supply'] = "Strong supply expansion"
        else:
            interpretations['supply'] = "Extreme supply expansion"
        
        # Demand interpretation (extended range)
        demand_score = scores.get('demand', 0)
        if demand_score <= -110:
            interpretations['demand'] = "Extreme demand restriction"
        elif demand_score <= -75:
            interpretations['demand'] = "Strong demand restriction"
        elif demand_score <= -25:
            interpretations['demand'] = "Demand reduction"
        elif demand_score <= 25:
            interpretations['demand'] = "Neutral demand impact"
        elif demand_score <= 75:
            interpretations['demand'] = "Demand increase"
        elif demand_score <= 110:
            interpretations['demand'] = "Strong demand expansion"
        else:
            interpretations['demand'] = "Extreme demand expansion"
        
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
        
        return f"{price_impact} {enforcement}"
    
    def _get_default_scores(self) -> Dict:
        """Return default scores when no similarities are available."""
        return {
            'supply': 0,
            'demand': 0,
            'policy_strength': 0,
            'interpretations': {
                'supply': "No supply impact detected",
                'demand': "No demand impact detected",
                'policy_strength': "No policy strength detected",
                'overall': "Unable to score document"
            }
        }