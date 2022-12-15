all: ocaml-build

ocaml-build:
	cd program_synthesis && \
	dune build && \
	cd ..

clean-ocaml-build:
	cd program_synthesis && \
	dune clean && \
	cd ..

pretty-repr-json:
	for f in $(DOMAIN)/representations/*.json; do \
	cat $$f | jq | sponge $$f; done

pretty-dsls-json:
	for f in $(DOMAIN)/dsls/*.json; do \
	cat $$f | jq | sponge $$f; done

pretty-json: pretty-programs-json pretty-dsls-json

clean-repr: 
	rm $(DOMAIN)/representations/*

clean-vis:
	rm $(DOMAIN)/visualizations/*
