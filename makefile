#############################
FILENAME := test.py
SCRIPT := test.py

check:
	ruff check --unsafe-fixes --fix

format:
	ruff format --config indent-width=2

script:
	python3 $(SCRIPT)

install:
	pip3 install -r requirements.txt

initialize:
	python3 -m pipreqs.pipreqs --encoding=utf8 --force