Cold-Start Generalization Evaluation
Experimental Design
We conducted rigorous cold-start evaluation following established protocols on the DAVIS dataset to assess model generalization under realistic scenarios:

Compound Cold-start: Unseen food-derived compounds

Protein Cold-start: Unseen protein targets

Dual Cold-start: Both compounds and proteins unseen

A structured grouping approach was employed, clustering compounds by structural similarity and proteins by phylogenetic family, with 20% of each group reserved as cold sets to ensure diversity and representativeness.

Comparative Performance
FoodAffNet_E was evaluated against strong baselines including GraphDTA, GLFA, GEFA, and FusionDTA:

Performance Metrics (CI ↑ / MSE ↓)

<img width="759" height="115" alt="image" src="https://github.com/user-attachments/assets/134f0955-c979-4983-a749-edaa0c859937" />

Key Findings
Superior Generalization: FoodAffNet_E consistently outperforms all baselines across all cold-start scenarios

Protein Cold-start: Achieves significantly higher CI (0.839) demonstrating enhanced capability with unseen targets

Compound Cold-start: Maintains robust performance (CI: 0.756) with novel food-derived compounds

Dual Cold-start: Sustains clear advantage (CI: 0.701) while achieving lower MSE in the most challenging scenario

Architecture Advantages
The performance gains are attributed to FoodAffNet_E's dual-stream, multi-scale fusion architecture:

Protein Stream: Integrates evolutionary and structural information for unseen target handling

Compound Stream: Captures hierarchical chemical semantics through convolutional and attention mechanisms

Multi-scale Fusion: Enables robust inference under high data sparsity through effective stream interaction

This architecture provides strong generalization capacity in practical cold-start regimes, making it particularly suitable for real-world applications involving novel entities.
