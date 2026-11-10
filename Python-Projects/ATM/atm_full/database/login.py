from database.databaseConnection import database_config, cursor
from database.utility import checkAccountStatus

class Login:

    def __init__(self):
        pass

    def login(self, account_no: int, password: str):
        try:
            # check whether account exists
            if checkAccountStatus(account_no=account_no):

                # get password from DB
                password_query = "SELECT PASSWORD FROM ACCOUNTS WHERE ACCOUNT_NO = %s;"
                cursor.execute(password_query, (account_no,))
                db_password = cursor.fetchone()

                if not db_password:
                    return False

                if db_password[0] == password:
                    return True
                else:
                    return False

            else:
                return False

        except Exception as e:
            return f"Something wrong in database/Login.login: {e}"
