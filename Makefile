all: ocaml-build

ocaml-build:
	cd program_synthesis && \
	dune build && \
	cd ..

clean-ocaml-build:
	cd program_synthesis && \
	dune clean && \
	cd ..

pretty-programs-json:
	for f in $(DOMAIN)/programs/*.json; do \
	cat $$f | jq | sponge $$f; done

pretty-dsls-json:
	for f in $(DOMAIN)/dsls/*.json; do \
	cat $$f | jq | sponge $$f; done

pretty-json: pretty-programs-json pretty-dsls-json

clean-executed-programs: 
	rm $(DOMAIN)/executed_programs/*

clean-vis:
	rm $(DOMAIN)/visualizations/*
