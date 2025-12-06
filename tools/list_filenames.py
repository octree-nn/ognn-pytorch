#!/usr/bin/env python3
import argparse
from pathlib import Path


def list_files_recursive(root_dir, output_file):
  # list all filenames in a directory recursively and save results to a file

  root = Path(root_dir).expanduser().resolve()
  with open(output_file, "w", encoding="utf-8") as f:
    for path in root.rglob("*"):
      if path.is_file():
        # write full path
        # f.write(str(path) + "\n")
        f.write(str(path.relative_to(root)) + "\n")
        # f.write(path.name + "\n")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Recursively list all files in a directory and save to a file."
  )
  parser.add_argument("root_dir", help="Root directory to scan")
  parser.add_argument("output_file", help="Path to output text file")

  args = parser.parse_args()
  list_files_recursive(args.root_dir, args.output_file)
