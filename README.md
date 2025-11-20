Multi-Task Clustering Federated Learning (MCFL)

This repository contains the implementation and research work for “Multi-Task Clustering for Personalized Federated Learning with Adaptive Weighting in Carbon Emission Prediction.”
The project introduces MCFL, a novel federated learning framework designed to improve personalization and accuracy in highly non-IID distributed environments.

Overview

MCFL integrates K-Means clustering, multi-task learning, and an adaptive expert layer weighting mechanism to enhance federated learning performance. The method groups clients with similar data distributions, builds personalized cluster models, and enables intelligent cross-cluster knowledge transfer.

Key Features
	•	Client clustering using K-Means
	•	Expert & personalization layer decomposition
	•	Adaptive weighting for cross-cluster feature sharing
	•	Faster convergence than FedAvg and CFL
	•	Improved performance on carbon emission prediction

Results Summary
	•	23% lower MSE than FedAvg
	•	6.8% better than CFL
	•	Stable convergence within 9 rounds
	•	Robust performance under 10, 50, and 100 heterogeneous client distributions
