import UserAccessor

class UserManager:
    """
        User Manager keeps track of the users in the system
    """
    
    def __init__(self, user_accessor: UserAccessor):
        self.admins = {"admins": []}
        self.users = {"users": []}
        self.user_accessor = user_accessor

    def add_admin(self, username):
        self.admins["admins"].append(username)
    
    def add_user(self, username):
        usernames = self.user_accessor.get_all_users()
        self.users["users"].append(usernames)

    def get_remote_username(self, req):
        return req.get("Username")
    
    def is_system_admin(self, username):
        return username in self.admins["admins"]