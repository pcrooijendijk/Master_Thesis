from typing import List

class UserAccessor:
    """
        User Accessor keeps track of users
    """
    def __init__(self):
        self.users = {"users": []}
    
    def add_user(self, username: str) -> None:
        self.users["users"].append(username)

    def get_all_users(self) -> dict:
        return self.users

    def get_user(self, username: str) -> bool:
        return username if username in self.users["users"] else None

class UserManager:
    """
        User Manager keeps track of the users in the system
    """
    
    def __init__(self, user_accessor: UserAccessor):
        self.admins = {"admins": []}
        self.users = {"users": []}
        self.user_accessor = user_accessor

    def add_admin(self, username: str) -> None:
        self.admins["admins"].append(username)
    
    def add_user(self, username: str) -> None:
        usernames = self.user_accessor.get_all_users()
        self.users["users"].append(usernames)

    def get_remote_username(self, req: dict) -> str:
        return req.get("Username")
    
    def is_system_admin(self, username: str) -> bool:
        return username in self.admins["admins"]
    
class UserPermissionsEntity:
    """
        User Permissions Entity keeps track of user permissions
    """

    def __init__(self, space_permissions: List):
        self.space_permissions = space_permissions
    
    def get_space_permissions(self) -> List:
        return self.space_permissions

    def set_space_permissions(self, new_space_permissions: List) -> None:
        self.space_permissions = new_space_permissions