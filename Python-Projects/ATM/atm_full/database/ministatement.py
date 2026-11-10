from database.databaseConnection import cursor, database_config
from database.utility import checkAccountStatus

# creating class for ministatement
class Ministatement:
    def __init__(self):
        pass
    def get_ministatement(self, account_no:int):
        try:
            # check account is exists or not in database
            if checkAccountStatus(account_no= account_no):
                # get last 5 transactions from database
                get_transactions_query = """SELECT * FROM TRANSACTIONS
                                            WHERE ACCOUNT_NUMBER = %s
                                            ORDER BY TIMESTAMP DESC
                                            LIMIT 10;"""
                cursor.execute(get_transactions_query, (account_no,))
                transactions = cursor.fetchall()
                return transactions
            else:
                return "Account number does not exist!"
        except Exception as e:
            return f"Something wrong in database/ministatement.get_ministatement: {e}"
