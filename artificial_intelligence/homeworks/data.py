from __future__ import annotations


try:
	import dotenv
except ImportError:
	dotenv = None

if dotenv is not None:
	dotenv.load_dotenv(override = True)

import datasets


data = datasets.load_dataset("ailsntua/QEvasion")
