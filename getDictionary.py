import glob
import pandas as pd

ans = []
for filename in glob.glob('**/*.txt', recursive=True):
  raw = open(filename, "r").read().split()[1:-2]
  for word in raw:
    ans.append(word)
dictionary = set(ans)

with open('dictionary.txt','w') as destfile:
    for word in dictionary:
        destfile.write('%s\n' % word)

print('finished')
