# Old reports
In this folder you will find anonymized report examples from 2025.

## Summary

### Report: Group 1
- **Project type:** Multi-class image classification of star constellations using a custom CNN.
- **Experiments performed:** Six iterative model versions; simple baseline; addition of a background class; handling class imbalance; data augmentation; dropout; weight decay; early stopping; final test-set comparison.
- **Extensions beyond standard:** None identified. The work is a systematic improvement of a conventional CNN classifier.

### Report: Group 2
- **Project type:** Multi-class skin-lesion image classification using transfer learning.
- **Experiments performed:** Frozen-classifier baseline and ResNet-50 fine-tuning; balanced-batch sampling versus weighted cross-entropy versus focal-loss variants; data augmentation; dropout and L2 weight decay; warm-up and cosine learning-rate schedules; confusion-matrix and Grad-CAM analysis.
- **Extensions beyond standard:** A comparatively thorough hypothesis-driven study of alternative class-imbalance treatments, but no new method proposed.

### Report: Group 3
- **Project type:** Joint deepfake image classification and pixel-level tamper segmentation.
- **Experiments performed:** ResNet-50 three-class classifier with focal loss; U-Net-style segmenter with a ResNet-50 encoder and hybrid segmentation loss; fixed data-subset experiments; separate classification and segmentation evaluation; comparison with the large SIDA-7B benchmark on F1, Dice, IoU, parameter count, and localization precision.
- **Extensions beyond standard:** A lightweight dual-pipeline design tested as an alternative to a large multimodal model; the classification and localization tasks are deliberately decoupled to avoid multi-task interference.

### Report: Group 4
- **Project type:** Face verification using Siamese metric-learning networks.
- **Experiments performed:** MobileNetV2 with triplet loss versus ResNet-18 with contrastive loss; augmentation and regularization variants; partial fine-tuning; verification accuracy and AUC comparisons; t-SNE embedding analysis, misclassified-pair inspection, and Grad-CAM visualization.
- **Extensions beyond standard:** Comparative study of two lightweight backbone/loss paradigms, supplemented by embedding-space and explainability analysis.

### Report: Group 5
- **Project type:** Highly imbalanced melanoma image classification, plus patient-level lesion-risk analysis.
- **Experiments performed:** Small baseline CNN versus a batch-normalized/global-average-pooling CNN; class-weighted loss; malignant-class oversampling; extensive geometric and photometric augmentation; lesion-level evaluation; two patient-level “ugly duckling” strategies using feature-space outliers and malignancy probabilities.
- **Extensions beyond standard:** Patient-level testing of the ugly-duckling hypothesis beyond ordinary lesion classification.

### Report: Group 6
- **Project type:** Mushroom-species image classification using MobileNetV2 transfer learning.
- **Experiments performed:** Replication of a four-species baseline; a four-species visually similar dataset; a 12-species scalability test; frozen versus full-network fine-tuning; learning-rate and regularization adjustments; augmentation; per-class error and training-stability analysis.
- **Extensions beyond standard:** Explicit replication-and-extension study testing whether a published lightweight approach survives increased visual similarity and class-count scaling.

### Report: Group 7
- **Project type:** Lightweight YOLO-style grid classification for football-player and football recognition, emphasizing a rare small-object class.
- **Experiments performed:** Minimal YOLO-inspired baseline; class-weighted loss; random oversampling; hybrid oversampling/undersampling; learning-rate range testing and cyclic learning rates; data augmentation; hard-negative mining; player, ball, and overall accuracy comparisons.
- **Extensions beyond standard:** Custom hybrid sampling plus hard-negative-mining pipeline tested as a lightweight alternative to heavier multi-scale detectors.

### Report: Group 8
- **Project type:** Human keypoint detection formulated as image-to-heatmap segmentation.
- **Experiments performed:** VGG16 encoder-decoder baseline with MSE; Gaussian target peaks; addition of U-Net skip connections; batch normalization; weighted cross-entropy versus Dice loss; L2 regularization; dataset increase from 500 to 2,000 images and larger batches; activation/gradient diagnostics and OKS evaluation.
- **Extensions beyond standard:** None identified. The project is an iterative architecture/loss study for a standard heatmap-based pose-estimation formulation.

### Report: Group 9
- **Project type:** Household-waste image classification using ResNet-18 transfer learning.
- **Experiments performed:** Initial and dataset-corrected baselines; FAISS-based near-duplicate detection/removal; normalization and augmentation; weight decay; dropout; alternative learning-rate schedules; class-level and validation comparisons.
- **Extensions beyond standard:** Dataset-integrity investigation showing how duplicates and labeling problems inflate validation performance.

### Report: Group 10
- **Project type:** Multiclass brain-tumor segmentation in MRI.
- **Experiments performed:** Direct comparison of 2D, 2.5D, and 3D U-Net architectures; different amounts of spatial/volumetric context; Dice and IoU by tumor subregion; computational-memory and optimization-stability comparison.
- **Extensions beyond standard:** Controlled research question testing whether additional volumetric context justifies its computational cost; the 2.5D design is evaluated as a practical middle ground.

### Report: Group 11
- **Project type:** Food-image classification under an intentionally low-data regime.
- **Experiments performed:** Full-data preliminary run; frozen ResNet-18 baseline using only 20 images per class; early stopping; dropout; data augmentation; progressive unfreezing/selective fine-tuning; validation and generalization comparisons.
- **Extensions beyond standard:** Controlled low-data simulation used to study regularization and fine-tuning behavior; no new model proposed.

### Report: Group 12
- **Project type:** Dog-breed image classification with auxiliary localization, landmark prediction, and segmentation.
- **Experiments performed:** Single-task baselines for classification and each auxiliary task; hard-parameter-sharing multi-task networks; different task-loss aggregation/weighting strategies; task-gradient relationship analysis; soft-parameter-sharing cross-stitch networks; classification, localization, landmark, and segmentation comparisons.
- **Extensions beyond standard:** Strong research-like multi-task study testing whether auxiliary visual tasks improve breed classification, including gradient analysis and cross-stitch soft sharing.

### Report: Group 13
- **Project type:** Fine-grained dog-breed image classification focused on weak/misclassified classes.
- **Experiments performed:** ResNet-50 baseline; targeted augmentation; bounding-box cropping; ResNet-50/Xception feature-level fusion; class-specific error analysis; saliency inspection; aggregate and weak-class accuracy comparison.
- **Extensions beyond standard:** Targeted weak-class hypothesis and feature-fusion model designed to improve visually similar breeds rather than only optimizing aggregate accuracy.

### Report: Group 14
- **Project type:** Joint ingredient multi-label classification and semantic food segmentation.
- **Experiments performed:** Dedicated classification baseline; dedicated segmentation baseline; shared-encoder multi-task model; comparison of single-task and joint training; task-loss weighting and negative-transfer analysis.
- **Extensions beyond standard:** Hypothesis-driven multi-task experiment testing whether a shared ResNet-50 encoder benefits both ingredient recognition and segmentation; observed negative transfer is a central result.

### Report: Group 15
- **Project type:** Multi-class fruit image classification.
- **Experiments performed:** CNN-from-scratch baseline; data augmentation; ResNet-50 transfer learning; custom soft cost-sensitive loss aimed at selected confusions; partial unfreezing with discriminative learning rates; baseline/final performance comparison.
- **Extensions beyond standard:** Custom soft cost-sensitive loss designed to penalize particular class confusions.

### Report: Group 16
- **Project type:** Synthetic-to-real domain adaptation for car object detection.
- **Experiments performed:** YOLO detector trained on synthetic data as the baseline; CycleGAN image-level adaptation; DANN feature-level domain alignment; teacher-student self-training with pseudo-labels; cross-method comparison on real target-domain images.
- **Extensions beyond standard:** Comparative research study of three fundamentally different unsupervised/semi-supervised domain-adaptation mechanisms.

### Report: Group 17
- **Project type:** Plant-disease object detection with YOLO-family models.
- **Experiments performed:** Anchor-based YOLOv5s/YOLOv5m versus anchor-free YOLOX-s; model-size comparison; CBAM attention added to YOLOv5; standard NMS versus Soft-NMS; training and optimization variants; minority/small-lesion detection evaluation.
- **Extensions beyond standard:** Mechanism-driven comparison across detector paradigm, attention enhancement, and post-processing; CBAM integration and Soft-NMS are tested as targeted enhancements.

### Report: Group 18
- **Project type:** Ovarian segmentation in T2-weighted MRI.
- **Experiments performed:** Standard U-Net; Attention U-Net; ResNet-34 transfer-learning model; min-max normalization versus two intensity-based preprocessing methods; full-slice versus unilateral masking; cross-validation and volumetric Dice evaluation.
- **Extensions beyond standard:** Unilateral Masking strategy designed to resolve contradictory supervision when only one of two ovaries is annotated; this geometric-label hypothesis is the main research contribution.

### Report: Group 19
- **Project type:** Retinal blood-vessel semantic segmentation.
- **Experiments performed:** Baseline U-Net; encoder/decoder simplification ablations; removal of skip connections; bottleneck changes; custom Swin U-Net; Attention U-Net with ASPP; Dice versus Focal Tversky loss; augmentation and dropout comparisons.
- **Extensions beyond standard:** Systematic component-necessity ablation, including transformer and attention/ASPP variants, rather than simply adding model complexity.

### Report: Group 20
- **Project type:** Fine-grained dog-breed image classification.
- **Experiments performed:** CNN baseline versus Vision Transformer; TransFG-inspired ViT variant; comparison of standard CLS-token use with soft attention-weighted patch pooling; accuracy and interpretability analysis.
- **Extensions beyond standard:** Differentiable soft attention pooling proposed as an alternative to TransFG’s hard top-k patch selection.

### Report: Group 21
- **Project type:** Four-class brain-tumor MRI classification.
- **Experiments performed:** CNN-from-scratch baseline versus ImageNet-pretrained ResNet-50; augmentation; class weighting; multilayer classification head with dropout; label smoothing; adaptive learning-rate reduction and early stopping; per-class evaluation.
- **Extensions beyond standard:** None identified. The work compares a standard baseline with a conventional transfer-learning pipeline.

### Report: Group 22
- **Project type:** Histopathology tumor-patch classification with cross-center stain/domain adaptation.
- **Experiments performed:** Source-domain classifier tested in-domain and cross-domain; baseline CycleGAN followed by batch normalization, LeakyReLU, and U-Net skip-connection variants; cycle-consistency and FID comparisons; classification before/after adaptation; Grad-CAM/Grad-CAM++ analysis; reconstruction-attention correlation experiment.
- **Extensions beyond standard:** Research-like semantic-preservation test comparing attention maps before and after CycleGAN round-trip reconstruction, demonstrating that visual fidelity need not preserve diagnostic reasoning.

### Report: Group 23
- **Project type:** Animal-footprint species image classification.
- **Experiments performed:** ResNet-18 transfer-learning baseline; augmentation; label smoothing; Adam versus SGD; multiple learning-rate schedules; neuron dropout and channel-level DropBlock; class-imbalance handling; final Grad-CAM best/worst-case analysis.
- **Extensions beyond standard:** Bias investigation using Grad-CAM to test whether the model relies on image aspect ratio rather than footprint morphology.

### Report: Group 24
- **Project type:** Building semantic segmentation in satellite imagery.
- **Experiments performed:** U-Net with pretrained ResNet-34 encoder; learning-rate finder and tuning; normalization; optimizer/momentum tests; gradient diagnostics; Grad-CAM; BCE+Dice versus boundary-aware loss; augmentation; random versus building-centered cropping; hard-negative sampling; IoU/error-heatmap comparison.
- **Extensions beyond standard:** Boundary-aware loss and building-focused sampling are tested; the main research result is that sampling/data presentation mattered more than architectural or loss complexity.

### Report: Group 25
- **Project type:** Weakly supervised 3D brain-aneurysm detection/segmentation from volumetric CT.
- **Experiments performed:** 3D U-Net baseline; attention gates; Gaussian heatmap targets; decaying heatmap width; classification and coordinate-regression auxiliary heads; background-suppression loss; curriculum semimetric Dice loss; uncertainty-weighted multi-task loss; ablations disabling attention, auxiliary heads, background loss, and curriculum learning; patch-sampling ratio tests.
- **Extensions beyond standard:** Substantial custom method combining curriculum semimetric Dice, background-aware regularization, Gaussian weak labels, attention, and uncertainty-weighted multi-task learning.

### Report: Group 26
- **Project type:** Multi-class sports image classification.
- **Experiments performed:** Basic ResNet-18 transfer-learning baseline versus optimized pipeline with augmentation, Mixup, weight decay, scheduled learning rate, and weighted sampling; tests using 100%, 10%, and 5% of training data; overall and weakest-class comparisons.
- **Extensions beyond standard:** Controlled data-scarcity robustness study; methods themselves are standard.

### Report: Group 27
- **Project type:** Facial age regression with predictive uncertainty.
- **Experiments performed:** GoogLeNet Gaussian-NLL regression baseline predicting mean and variance; weighted age sampling; Gaussian blur; augmentation; variance/calibration experiments; ensembles of two to six models; uniform versus weighted ensemble forwarding; decade-wise MAE; Grad-CAM versus Grad-CAM++.
- **Extensions beyond standard:** Heteroscedastic uncertainty prediction and confidence-interval evaluation, plus hypothesis-driven ensemble weighting and calibration analysis.

### Report: Group 28
- **Project type:** Vision-language image classification and retrieval using prompt adaptation for MobileCLIP2.
- **Experiments performed:** Zero-shot baseline; CoOp-style learnable text prompts across MobileCLIP2 sizes and multiple datasets; learning-rate and prompt-length tuning; visual prompting; combined/multitask classification-retrieval training; validation/test evaluation; efficiency and model-size comparisons.
- **Extensions beyond standard:** Research-like adaptation of context optimization to resource-constrained MobileCLIP2, including visual prompts and a tested multitask hypothesis that was not supported.

### Report: Group 29
- **Project type:** Four-class brain-tumor MRI classification with hierarchical multi-task representation learning.
- **Experiments performed:** Transfer-learning baselines using Swin Transformer, ResNeXt, and MobileNet; data augmentation, label smoothing, weighted sampling, and hyperparameter tuning; hierarchical U-Net jointly performing reconstruction, binary tumor detection, and subtype classification; raw versus duplicate-cleaned datasets; UMAP distribution-shift analysis.
- **Extensions beyond standard:** Custom hierarchical multi-task U-Net with adaptive EMA loss normalization; dataset-shift hypothesis tested through latent-space analysis and removal of 1,113 duplicates.

### Report: Group 30
- **Project type:** Earthworm genus image classification and individual-specimen verification.
- **Experiments performed:** ResNet-50, ConvNeXt, and Swin Transformer comparison for five genera; segmented/cropped images with different amounts of background; augmentation and balancing; ResNet-50 triplet-loss embeddings for same-individual identification; t-SNE embedding analysis.
- **Extensions beyond standard:** Adds individual-earthworm metric learning beyond genus classification and explicitly tests the hypothesis that background area affects classifier performance.

### Report: Group 33
- **Project type:** Multi-output time-series classification of main crops and autumn vegetation from satellite data.
- **Experiments performed:** CNN on temporal-difference matrices plus GRUs on absolute Sentinel-1/2 series; eight integration modes for model sub-parts; hierarchical use of main-crop predictions for autumn-vegetation prediction; semi-balanced sampling; conservative noise augmentation; task-loss weighting; spatial holdout evaluation on unseen geographic areas.
- **Extensions beyond standard:** Custom hybrid CNN-GRU, hierarchical dual-task architecture and explicit spatial-transferability experiment for an underexplored Danish catch-crop setting.

### Report: Group 34
- **Project type:** Clinical image classification on high-dimensional Imaging Mass Cytometry data.
- **Experiments performed:** Four channel-processing/model architectures—learned channel adapter with ResNet-18, biologically grouped channels, modified-channel CNNs, and SPCA/RBF-SVM feature extraction; progressive unfreezing; oversampling; several clinical targets and pre/post-treatment subsets; patient-level five-fold cross-validation; UMAP and confusion-matrix analysis.
- **Extensions beyond standard:** Biologically grouped multichannel processing and comparison of end-to-end CNNs with an interpretable SPCA/SVM pipeline for spatial immunophenotyping.

### Report: Group 35
- **Project type:** Three-class monkeypox/skin-condition image classification.
- **Experiments performed:** Custom CNN-from-scratch baseline versus VGG16 transfer learning; augmentation; global-average-pooling classification head; dropout; Adam versus SGD considerations; learning-rate reduction and early stopping; per-class metrics; Grad-CAM analysis of correct and incorrect predictions.
- **Extensions beyond standard:** None identified. Grad-CAM is used to assess whether improved performance reflects meaningful lesion features.

### Report: Group 37
- **Project type:** Fast neural style transfer.
- **Experiments performed:** Standard Transformer Net baseline versus three progressively lightweight variants replacing different convolutions with depthwise-separable convolutions; perceptual content/style/total-variation loss training; parameter count, CPU latency, convergence, visual quality, and artifact comparison.
- **Extensions beyond standard:** Structured architecture-efficiency study testing how far depthwise-separable convolution can replace standard convolution while preserving style quality.

### Report: Group 38
- **Project type:** Multimodal skin-lesion image classification using images plus patient metadata.
- **Experiments performed:** Seven iterations establishing a ResNet-50 image-only baseline; SGD versus Adam; balanced sampling; augmentation and test-time augmentation; learning-rate scheduling and tentative fine-tuning; mid-level feature fusion versus FiLM conditioning; FiLM insertion-point comparison; metadata distribution and class-level analysis.
- **Extensions beyond standard:** Research-like comparison of metadata-fusion mechanisms, including FiLM at different network stages, testing whether clinical metadata improves robustness over an image-only baseline.

### Report: Group 42
- **Project type:** Low-label brain-tumor image classification using contrastive representation learning.
- **Experiments performed:** Reconstructed CNN baseline; supervised contrastive loss versus unsupervised NT-Xent across decreasing labeled-data fractions; ResNet-50 SimCLR-style model; temperature, learning rate, epochs, and subset-size experiments; four augmentation families; prototypical classification; t-SNE and Interactive-CAM analysis; repeated-run robustness test.
- **Extensions beyond standard:** Strong research-like test of label dependence in SupCon versus NT-Xent and of medical versus generic augmentations under extreme data scarcity.

### Report: Group 43
- **Project type:** Self-supervised image representation learning with SimSiam, compared with supervised classification.
- **Experiments performed:** Matched supervised and SimSiam baselines; weak versus strong augmentations; encoder width/depth scaling; residual connections; learning-rate swaps; batch-normalization removal/reintroduction; projector/predictor simplification; training-duration comparison; kNN accuracy and feature-standard-deviation collapse diagnostics.
- **Extensions beyond standard:** Extensive hypothesis-driven study testing whether findings from SimCLR transfer to SimSiam, including a mechanistic hypothesis that batch normalization prevents representation collapse.

### Report: Group 44
- **Project type:** Hierarchical fine-grained vehicle image classification.
- **Experiments performed:** Frozen and fine-tuned ResNet-50 baselines; flat single-head, two-head, and three-head make/type/model architectures; augmentation, class weighting, dropout, weight decay, and scheduling; simultaneous versus curriculum multi-task training; hierarchical label smoothing; two-head/three-head ablation; test-time augmentation.
- **Extensions beyond standard:** Custom hierarchical multi-task training with coarse-to-fine curriculum and taxonomy-aware label smoothing; explicit ablation supports the Type head as a semantic bridge.

### Report: Group 49
- **Project type:** Plant-disease image classification with cross-domain generalization.
- **Experiments performed:** AlexNet transfer-learning baseline trained on controlled PlantVillage images; testing on PlantVillage, PlantDoc, and FieldPlantVillage; addition of real-world PlantDoc images to training; further fine-tuning; within-domain versus external-domain accuracy comparison.
- **Extensions beyond standard:** Explicit domain-generalization hypothesis testing whether adding uncontrolled field images improves performance on external datasets.

### Report: Group 50
- **Project type:** Medical image segmentation with self-supervised masked pretraining.
- **Experiments performed:** Fully supervised U-Net baseline versus masked-image reconstruction pretraining followed by segmentation fine-tuning; different labeled/unlabeled data regimes on thyroid-ultrasound and panoramic X-ray datasets; standard U-Net versus lightweight NAC/scAG U-Net variants; Dice-based comparisons.
- **Extensions beyond standard:** Novel/research-like test of masked image modeling directly with CNN/U-Net architectures, challenging the view that masked pretraining primarily benefits Transformers.
