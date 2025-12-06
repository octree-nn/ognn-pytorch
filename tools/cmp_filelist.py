

def get_filenames(filelist):
  r''' Gets filenames from a filelist.
  '''

  with open(filelist, 'r') as fid:
    lines = fid.readlines()
  filenames = [line.split()[0] for line in lines]
  return filenames


def compare_filelists(filelist1, filelist2):
  r''' Compares two filelists.
  '''

  filenames1 = get_filenames(filelist1)
  filenames2 = get_filenames(filelist2)

  set1 = set(filenames1)
  set2 = set(filenames2)

  only_in_1 = set1 - set2
  only_in_2 = set2 - set1

  if only_in_1:
    print(f'Files only in {filelist1}:')
    for filename in sorted(only_in_1):
      print(f'  {filename}')
  else:
    print(f'No files only in {filelist1}.')

  if only_in_2:
    print(f'Files only in {filelist2}:')
    for filename in sorted(only_in_2):
      print(f'  {filename}')
  else:
    print(f'No files only in {filelist2}.')


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(
      description="Compare two filelists and report differences."
  )
  parser.add_argument("filelist1", help="First filelist to compare")
  parser.add_argument("filelist2", help="Second filelist to compare")

  args = parser.parse_args()
  compare_filelists(args.filelist1, args.filelist2)
