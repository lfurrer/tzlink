# Makefile for downloading large-ish files not included in the repo.

# Default target: initialise data/*
init: embeddings ncbi-disease

# Optional data: requires authentication
optional: bpe embeddings-chiu

# Run target: train a model and evaluate it.
run:
	python3 -m src.rank -t -p -r


# Intermediate targets: specific paths.
embeddings: data/embeddings/wvec_50_haodi-li-et-al.bin

bpe: $(addprefix data/embeddings/bpe_,abstract10000model vectors_10000_50_w2v.txt)

embeddings-chiu: data/embeddings/wvec_200_win-30_chiu-et-al.bin


ncbi-disease: nd-corpus nd-terminology

nd-corpus: $(addprefix data/ncbi-disease/NCBI,$(addsuffix set_corpus.txt,train test develop))

nd-terminology: data/ncbi-disease/CTD_diseases.tsv


# Directories.
data data/ncbi-disease data/embeddings:
	mkdir -p $@

# Leaf targets: download commands.
data/embeddings/wvec_50_haodi-li-et-al.bin: | data/embeddings
	wget -O $@ https://github.com/wglassly/cnnormaliztion/raw/master/src/embeddings/vec_50.bin

data/ncbi-disease/%.txt: | data/ncbi-disease
	wget -O - https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/$(subst .txt,.zip,$(@F)) | funzip > $@

data/ncbi-disease/CTD_diseases.tsv: | data/ncbi-disease
	wget -O - https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/download/DNorm/DNorm-0.0.7.tgz | tar -xzOf - DNorm-0.0.7/data/$(@F) > $@

# Optional leaf targets.
data/embeddings/bpe%: | data/embeddings
	scp evex.utu.fi:/home/lhchan/glove/selftrained_bpe_model/$(subst bpe_,,$(@F)) $@

data/embeddings/wvec_200_win-30_chiu-et-al.bin: | data/embeddings
	@# File ID given in https://github.com/cambridgeltl/BioNLP-2016 (README.md)
	python3 src/util/gdrive.py 0BzMCqpcgEJgiUWs0ZnU0NlFTam8 | tar -xzOf - bio_nlp_vec/PubMed-shuffle-win-30.bin > $@
