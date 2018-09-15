def test(name,age,city,language):
    print(name,age,city,language)

dic = {'name': 'dokelung', 'age': 27, 'city': 'Taipei', 'language': 'Python'}
test(**dic)