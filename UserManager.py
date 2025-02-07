class UserManager:
    """
        User Manager keeps track of the users in the system
    """
    
    def __init__(self):
        self.admins = {"admin", "user1"}
        self.users = {"user1", "user2"}

    def get_remote_username(self, req):
        return req.get("Username")
    
    def is_system_admin(self, username):
        return username in self.admins