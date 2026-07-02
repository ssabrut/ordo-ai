.PHONY: help install test test-unit test-integration \
	generate-dataset backfill-annotations augment-entities split-disfluency \
	voice-session

RUN := poetry run

help:
	@echo "install              - install deps via poetry"
	@echo "test                 - run full pytest suite"
	@echo "test-unit            - run tests/unit only"
	@echo "test-integration     - run tests/integration only"
	@echo "generate-dataset     - generate synthetic ordering dataset (MLX-LM)"
	@echo "backfill-annotations - backfill disfluency/NER annotations on intent_dataset.jsonl"
	@echo "augment-entities     - augment dataset via entity substitution"
	@echo "split-disfluency     - split disfluency repair dataset into train/val/test"
	@echo "voice-session        - run local voice session against the LangGraph pipeline"

install:
	poetry install

test:
	$(RUN) python -m pytest tests/ -v

test-unit:
	$(RUN) python -m pytest tests/unit -v

test-integration:
	$(RUN) python -m pytest tests/integration -v

generate-dataset:
	$(RUN) python scripts/generate_ordering_dataset.py --count 50

backfill-annotations:
	$(RUN) python scripts/backfill_dataset_annotations.py

augment-entities:
	$(RUN) python scripts/augment_entity_substitution.py

split-disfluency:
	$(RUN) python scripts/split_disfluency_repair_dataset.py

voice-session:
	$(RUN) python scripts/run_voice_session.py
