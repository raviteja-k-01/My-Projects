import addition
import subtraction
import multiplication
import division
import mod_division
import exponentiation
operations= ('1.addition','2.subtraction','3.multiplication','4.division','5.modulus','6.exponentiation','7.exit')

if __name__=='__main__':
    a,b=map(int,input("Enter two numbers: ").split(','))
    print('operations are:',*operations,sep='\n')
    while True:
            choice=int(input("Enter your choice of operation: "))  
            if choice==1:
                print("Result:",addition.add(a,b))
            elif choice==2:
                print("Result:",subtraction.sub(a,b))
            elif choice==3:
                print("Result:",multiplication.mul(a,b))
            elif choice==4:
                print("Result:",division.div(a,b))
            elif choice==5:
                print("Result:",mod_division.mod_div(a,b))
            elif choice==6:
                print("Result:",exponentiation.exp(a,b))
            elif choice==7:
                print("exiting")
                break
            else:
            #this will break the loop if the choice is invalid
               print("Invalid choice")

