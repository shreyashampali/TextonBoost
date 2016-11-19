1. TextonBoost algorithm (https://webdocs.cs.ualberta.ca/~btap/readings/papers/CRF_TextonBoost_ECCV2006.pdf) implementation in matlab
2. Currently works for single class classification. Multi-class classification not supported
3. Usage :
	1. gentleBoost.m - Classifier training
	2. classifier.m  - Image classification using the classifier trained in gentleBoost.m
	3. generateFilters.m, getTextonMap.m, getFeatureVectorsinImage.m, generateTextonFilter.m and alphaExpansion.m - supporting codes for above two programs
4. Input images : Training and testing images for a particular class should be stored in this directory structure format : "<class_name>/training" and "<class_name>/testing"