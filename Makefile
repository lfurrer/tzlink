# Makefile for downloading large blobs not included in the repo.

# Default target: initialise data/*
init: embeddings ncbi-disease


# Intermediate targets: specific paths.
embeddings: data/wvec_50_haodi-li-et-al.bin

ncbi-disease: $(addprefix data/ncbi-disease/NCBI,$(addsuffix set_corpus.txt,train test develop))


# Leaf targets: download commands.
data/wvec_50_haodi-li-et-al.bin:
	mkdir -p $(@D)
	wget https://github.com/wglassly/cnnormaliztion/raw/master/src/embeddings/vec_50.bin -O $@

data/ncbi-disease/%.txt:
	mkdir -p $(@D)
	wget https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/$(subst .txt,.zip,$(@F)) -O - | funzip > $@
