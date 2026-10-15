#!/usr/bin/env python
"""Run and administer the project, Django style:  python manage.py <command>

    python manage.py migrate            tables and customers in the database from .env
    python manage.py createsuperuser    the first admin account
    python manage.py runserver          the API and the dashboard; Ctrl+C stops both

python manage.py --help lists every command.
"""

import sys
from pathlib import Path

# Works before `pip install -e .` too.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from recommender.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
