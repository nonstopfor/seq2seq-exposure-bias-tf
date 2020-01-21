def test_args(first, *args):
   print('Required argument: ', first)
   print(type(args))
   for v in args:
      print('Optional argument: ', v)
def test1(first,second,third):
   print(first,second,third)
test_args(1, [2, 3, 4])
x=(1,2,3)

#print(type(*x))
test1(*x)
class A():
   def __init__(self):
      pass

a=A()
a.x=3
print(a.x)