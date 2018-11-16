import re
import string

line = 'qq'
while(line):
    line = re.sub('['+string.punctuation+']', ' ', line)

    print(line)
    line = input()


