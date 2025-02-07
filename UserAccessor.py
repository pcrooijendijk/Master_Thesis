class UserAccessor:
    """
        User Accessor keeps track of users
    """
    def __init__(self):
        self.users = {"user1", "user2", "admin"}
    
    def get_user(self, username):
        return username if username in self.users else None