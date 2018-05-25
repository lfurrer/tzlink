# Makefile for downloading large-ish files not included in the repo.

# Default target: initialise data/*
init: embeddings ncbi-disease

# Run target: train a model and evaluate it.
run:
	python3 -m src.rank -d ncbi-disease -t -p


# Intermediate targets: specific paths.
embeddings: data/wvec_50_haodi-li-et-al.bin


ncbi-disease: nd-corpus nd-terminology

nd-corpus: $(addprefix data/ncbi-disease/NCBI,$(addsuffix set_corpus.txt,train test develop))

nd-terminology: data/ncbi-disease/CTD_diseases.tsv


# Directories.
data data/ncbi-disease:
	mkdir -p $@

# Leaf targets: download commands.
data/wvec_50_haodi-li-et-al.bin: | data
	wget -O $@ https://github.com/wglassly/cnnormaliztion/raw/master/src/embeddings/vec_50.bin

data/ncbi-disease/%.txt: | data/ncbi-disease
	wget -O - https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/$(subst .txt,.zip,$(@F)) | funzip > $@

data/ncbi-disease/CTD_diseases.tsv: | data/ncbi-disease
	wget -O - https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/download/DNorm/DNorm-0.0.7.tgz | tar -xzOf - DNorm-0.0.7/data/$(@F) > $@
