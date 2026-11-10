from database.ministatement import Ministatement

# Mini statement function for Flask
def ministatement(account:int):
    try:
        ministatement_obj = Ministatement()
        transactions = ministatement_obj.get_ministatement(account_no=account)
        
        # If the returned value is a list of transactions, return it
        if isinstance(transactions, list):
            print(transactions)
            return transactions
        else:
            # If something went wrong (e.g., account does not exist), return empty list
            return []
    except Exception as e:
        print(f"Error in ministatement function: {e}")
        return []
    pass
print(ministatement(1234))
# Logout function for Flask
from flask import session, redirect
def logout_flask():
    session.clear()  # clear all session data
    return redirect("/login")
