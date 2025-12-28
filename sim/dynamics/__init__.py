"""
Dynamic Network Evolution Module
=================================
Implements dynamic social network evolution based on beliefs and interactions.

Key mechanisms:
1. Belief-Driven Rewiring: Agents adjust connections based on belief similarity
2. Echo Chamber Formation: Emergent homophily creates filter bubbles
3. Influence Dynamics: Dynamic influence based on credibility and position
4. Bridge Dissolution: Cross-cutting ties weaken under polarization
5. Triadic Closure: Friends of friends become friends

References:
- Centola, D. (2010). The spread of behavior in an online social network
- Bakshy, E., et al. (2015). Exposure to ideologically diverse news
- Del Vicario, M., et al. (2016). The spreading of misinformation online
- Bail, C. A., et al. (2018). Exposure to opposing views on social media
"""

from sim.dynamics.network_evolution import (
    NetworkEvolutionConfig,
    DynamicNetworkState,
    initialize_dynamic_network,
    update_network_structure,
    compute_rewiring_probabilities,
    detect_echo_chambers,
    measure_bridge_ties,
    compute_network_polarization,
)

from sim.dynamics.influence import (
    InfluenceConfig,
    InfluenceState,
    initialize_influence_state,
    compute_dynamic_influence,
    update_influence_scores,
    identify_superspreaders,
    compute_influence_flow,
)

__all__ = [
    "NetworkEvolutionConfig",
    "DynamicNetworkState",
    "initialize_dynamic_network",
    "update_network_structure",
    "compute_rewiring_probabilities",
    "detect_echo_chambers",
    "measure_bridge_ties",
    "compute_network_polarization",
    "InfluenceConfig",
    "InfluenceState",
    "initialize_influence_state",
    "compute_dynamic_influence",
    "update_influence_scores",
    "identify_superspreaders",
    "compute_influence_flow",
]
