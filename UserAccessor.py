class UserAccessor:
    """
        User Accessor keeps track of users
    """
    def __init__(self):
        self.users = {"users": []}
    
    def add_user(self, username):
        self.users["users"].append(username)

    def get_all_users(self):
        return self.users

    def get_user(self, username):
        return username if username in self.users["users"] else None