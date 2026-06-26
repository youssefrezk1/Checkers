# checkers/evaluation/forbidden_vocab.py
#
# Re-export shim — canonical definitions live in checkers.ontology.forbidden_vocab
# so that the runtime agent can import from checkers.ontology without pulling
# checkers.evaluation into sys.modules (Fix-5 isolation invariant).
#
# All existing evaluation-layer imports of ABSOLUTE_FORBIDDEN_VOCAB and
# CONTEXT_FORBIDDEN_VOCAB continue to work unchanged.

from checkers.ontology.forbidden_vocab import (  # noqa: F401
    ABSOLUTE_FORBIDDEN_VOCAB,
    CONTEXT_FORBIDDEN_VOCAB,
)
