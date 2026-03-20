# NTVGCL
NTVGCL: Natural Twin Views Graph Contrastive Learning forSelf-Supervised In-Vehicle Intrusion Detection
we propose Natural Twin Views Graph Contrastive Learning (NTVGCL), a selfsupervised in-vehicle intrusion detection method that achieves superior detection efficacy. NTVGCL employs a Siamese architecture and leverages the principle of temporal consistency to derive Natural Twin Views (NTVs) from temporally adjacent Message Interaction Graphs (MIGs). The deterministic nature of NTVs eliminates the need for hand-crafted augmentations and avoids the instability inherent in traditional unsupervised learning, thereby more effectively capturing latent and robust invariant representations of normal CAN traffic patterns. The resulting representations are processed by an anomaly discrimination module that utilizes an autoencoder to efficiently quantify deviations from normal patterns through reconstruction errors. Extensive experiments on four benchmark datasets indicate that NTVGCL consistently outperforms state-of-the-art models, achieving an AUC of 99% and an F1-score ranging from 94% to 99%. 

The HCRL folder contains the raw CAN data.
The *_graphs folder contains the processed graph data for each dataset.
The *_graphs_results folder contains the best trained models and inference results for each dataset.
*.py files are the Python code files for NTVGCL.
