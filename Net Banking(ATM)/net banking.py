#creating an ATM 
accounts={'abc':123,'xyz':456,'def':789} #login info
acc_details={'abc':['ravi',1000.0,[]],
             'xyz':['teja',1500.0,[]],
             'def':['varun',2000.0,[]]}
operations=('1.create_accnt','2.delete_accnt','3.deposit','4.withdrawal','5.transfer','6.check balance','7.history','8.exit')

def login(username,password):
    if username in accounts and accounts[username] == password:
        print(f"Login successful. Welcome, {acc_details[username][0]}!")
        return True
    else:
         return False
    
def create_new_accnt():
    new_username=input("enter new username : ")
    new_password=int(input('enter new password: '))
    name=input("enter a unique name: ")
    initial_deposit=int(input("enter initial deposit: "))
    if new_username in accounts:
        return "Account already exists"
    else:
        accounts[new_username] = new_password
        acc_details[new_username] = [name, initial_deposit,[]] 
        print(accounts)
        return f"Account created successfully for {name}."

def delete_accnt(username):
     if username in accounts:
          accounts.pop(username)
          print(accounts)
          return f"account:{username} deleted successfully"
     else:
          return "account not found"

def deposit(username,amount):
        if amount>0:
             acc_details[username][1]= acc_details[username][1]+amount
             acc_details[username][2].append(f"successfully deposited :{amount}")
             return f"updated balance:{acc_details[username][1]}"
        else:
             return "please enter valid amount"
        
def withdrawal(username,amount):
     if acc_details[username][1]>0 and amount<=acc_details[username][1]:
          acc_details[username][1]=acc_details[username][1]-amount
          acc_details[username][2].append(f"successful withdrawal :{amount}")
          return f"updated balance: {acc_details[username][1]}"
     else:
          return "insufficient amount"
     
def check_balance(username):
      return f"current balance is: {acc_details[username][1]}"
        
def transfer(username, amount):
    username_1 = input("enter the account to be transfered: ")
    if amount > 0:
        if username_1 == username:
            return "Cannot transfer to the same account"
        if username_1 in acc_details:
            if acc_details[username][1] >= amount:
                acc_details[username_1][1] += amount
                acc_details[username][1] -= amount
                acc_details[username][2].append(f"successfully transferred :-{amount} to {username_1}")
                acc_details[username_1][2].append(f"successfully received :+{amount} from {username}")
                return "amount transfered successfully"
            else:
                return "insufficient balance to transfer"
        else:
            return "no users found"
    else:
        return "invalid input"
   
def history(username):
      return acc_details[username][2]

if __name__=='__main__':
      username=input("enter the username of the acnt:")
      password=int(input("enter password : "))
      if login(username,password):
               while True:
                    print(*operations,sep='\n')
                    operation=input("select the operation that need to be performed:")
                    if operation=='1':                     
                         print(create_new_accnt())
                    elif operation=='2':
                         print(delete_accnt(username))
                    elif operation=='3':
                         amount=int(input("enter amount to deposit : "))
                         print(deposit(username,amount))
                    elif operation=='4':
                         amount=int(input("enter amount to withdraw : "))
                         print(withdrawal(username,amount))
                    elif operation=='5':
                         amount=int(input("enter amount to transfer: "))
                         print(transfer(username,amount))  
                    elif operation=='6':
                         print(check_balance(username))   
                    elif operation=='7':
                         print(history(username))
                    elif operation=='8':
                         print("exiting")
                         break
                    else:
                         print("invalid input,please try again")
      else:
          print("login failed")
                       
      
      

