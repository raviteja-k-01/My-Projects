operations=("1.adition","2.subtaction","3.multiplication","4.division","5.modulo","6.power","7.exit")
def add(a,b):
    return a+b
def sub(a,b):
    return a-b
def mul(a,b):
    return a*b
def div(a,b):
    try:
        return a/b
    except ZeroDivisionError:
        return "b is zero enter again"
def mod(a,b):
    return a%b
def power(a,b):
    return a**b    
    
if __name__=='__main__':
    a,b=map(int,input("enter two numbers: ").split(","))
    print(*operations,sep='\n')
    while True:
        choice=int(input("enter the operation to perform:"))
        if choice==1:
            print(f"solution is : {add(a,b)}")
        elif choice==2:
            print(f"solution is:{sub(a,b)}")
        elif choice==3:
            print(f"solution is:{mul(a,b)}")
        elif choice==4:
            print(f"solution is:{div(a,b):.2f}")
        elif choice==5:
            print(f"solution: {mod(a,b)}")  
        elif choice==6:
            print(f"solution is : {power(a,b)}")       
        elif choice==7:
            print("exiting")
            break              
