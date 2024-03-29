import re
import unicodedata
      
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize("NFD",s) 
                    if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    print("1 :",[s])
    s = re.sub(r"([.!?])", r" \1", s)
    print("2 :",[s])
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

while(1):
    test = input()
    print("3 :",[normalizeString(test)])
    print('*'*20)