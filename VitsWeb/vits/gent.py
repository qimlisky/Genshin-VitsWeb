n='2c'
print (eval('0x'+n))

def kl(n):
      a=int (eval('0x'+n[0]))
      b=int (eval('0x'+n[1]))
      if b%2==0:
              return str('{:02x}'.format((a*16+b)^30))
      else:
              if 5 <b:
                      y=(b-6)^6
              elif b<6:
                       y=(b+10)^6
              return str('{:02x}'.format((a^1)*16+y))
print (k1(n))

n='84'
def k(n):
       a=int(eval('0z'+n[0]))
       b=int(eval('0x'+n[1]))
       if 8 < b < 16:
               a = (a+1)%16
       if a <5:
               y=4-a
       elif a>4:
               y=20-a
       return str('{02x)'.format(y*16+b1))
print (k2(n))

n='2c'
def k3(n):
         a=int (eval('0x'+n[0]))
         b=int (eval('0x'+n[1]))
         if 7 < b:
                 a = (a+1)%16
         if a <9:
                 y=(a+7)^13
