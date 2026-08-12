"""Measurement layer for "does the evolved playbook actually help?".

Everything in here is pure: it reads artifacts that a run already wrote and
turns them into a verdict. No LLM calls, no GPU, no file mutation. The driver
that *produces* those artifacts is `scripts/verify_playbook_effect.py`; keeping
the two apart is what lets the decision rule be unit-tested against fixtures
instead of against a four-hour run.
"""
