from perm_utils import Role
from typing import List
from perm_utils.UserPermissionManagement import UserPermissionsResource
import json

class Users():
    def __init__(self, space_keys: List):
        self.space_keys = space_keys
        self.clients = []

        self.set_users()

    def set_users(self) -> None:
        # Initialize the users where they have per space the option to be admin and a list of permissions
        self.users = {
            "admin": {
                "id": 1,
                "space": [self.space_keys[0], self.space_keys[1]],
                "is_admin": True,
                "permissions": Role.get_role_permissions()["admin"]
            },
            "user1": {
                "id": 2,
                "space": [self.space_keys[1], self.space_keys[2]],
                "is_admin": False,
                "permissions": Role.get_role_permissions()["editor"]
            },
            "user2": {
                "id": 3,
                "space": [self.space_keys[2]],
                "is_admin": False,
                "permissions": Role.get_role_permissions()["viewer"]
            },
            "user3": {
                "id": 4,
                "space": [self.space_keys[3]],
                "is_admin": True,
                "permissions": Role.get_role_permissions()["editor"]
            },
            "user4": {
                "id": 5,
                "space": [self.space_keys[0]],
                "is_admin": False,
                "permissions": Role.get_role_permissions()["viewer"]
            },
            "user5": {
                "id": 6,
                "space": [self.space_keys[1], self.space_keys[3]],
                "is_admin": False,
                "permissions": Role.get_role_permissions()["editor"]
            },
            "user6": {
                "id": 7,
                "space": [self.space_keys[2]],
                "is_admin": False,
                "permissions": Role.get_role_permissions()["editor"]
            },
            "user7": {
                "id": 8,
                "space": [self.space_keys[0], self.space_keys[3]],
                "is_admin": True,
                "permissions": Role.get_role_permissions()["viewer"]
            },
            "user8": {
                "id": 9,
                "space": [self.space_keys[1]],
                "is_admin": False,
                "permissions": Role.get_role_permissions()["viewer"]
            },
            "user9": {
                "id": 10,
                "space": [self.space_keys[3]],
                "is_admin": False,
                "permissions": Role.get_role_permissions()["editor"]
            },
            "user10": {
                "id": 11,
                "space": self.space_keys,  # Full access to all spaces
                "is_admin": True,
                "permissions": Role.get_role_permissions()["admin"]
            }
        }

        self.users_names = list(self.users.keys()) # Make a list of the keys 

    def set_clients(self, user_permissions_resource: UserPermissionsResource) -> None:
        from fed_utils import Client
        for user in self.users:
            self.clients.append(
                Client(
                    client_id=self.users[user]['id'], 
                    name=user, 
                    user_permissions_resource=user_permissions_resource
                )
            )
    
    def get_users(self) -> dict: 
        return self.users

    def get_clients(self) -> List: 
        return self.clients