import path  # setup path
import argparse
from pathlib import Path
from sqlalchemy import create_engine
from db import create_db


def main():
    parser = argparse.ArgumentParser(description="Create a database")
    parser.add_argument("db", help="database uri")
    parser.add_argument("category_path", type=Path, help="path to the categories")
    parser.add_argument("pages_path", type=Path, help="path to the pages")
    parser.add_argument(
        "--store-content", action="store_true", help="store actual binary content"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="number of workers to use"
    )

    args = parser.parse_args()

    create_db(
        engine=create_engine(args.db),
        category_path=args.category_path,
        pages_path=args.pages_path,
        store_content=args.store_content,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
