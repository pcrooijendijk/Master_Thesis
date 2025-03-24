from perm_utils import Role
from typing import List

def make_users(space_keys: List) -> dict:
    # Initialize the users where they have per space the option to be admin and a list of permissions
    users = {
        "admin": {
            "id": 1, 
            "space": space_keys[0],
            "is_admin": True,
            "permissions": Role.get_role_permissions()["admin"]
        }, 
        "user1": {
            "id":2, 
            "space": space_keys[1],
            "is_admin": False,
            "permissions": Role.get_role_permissions()["editor"]
        }, 
        "user2": {
            "id":2, 
            "space": space_keys[2],
            "is_admin": False,
            "permissions": Role.get_role_permissions()["viewer"]
        }, 
        "user3": {
            "id":2, 
            "space": space_keys[3],
            "is_admin": False,
            "permissions": Role.get_role_permissions()["editor"]
        }, 
        "user4": {
            "id":2, 
            "space": space_keys[0],
            "is_admin": False,
            "permissions": Role.get_role_permissions()["viewer"]
        }, 
        "user5": {
            "id":2, 
            "space": space_keys[1],
            "is_admin": False,
            "permissions": Role.get_role_permissions()["editor"]
        }, 
        "user6": {
            "id":2, 
            "space": space_keys[2],
            "is_admin": True,
            "permissions": Role.get_role_permissions()["viewer"]
        }, 
        "user7": {
            "id":2, 
            "space": space_keys[3],
            "is_admin": False,
            "permissions": Role.get_role_permissions()["editor"]
        }, 
        "user8": {
            "id":2, 
            "space": space_keys[0],
            "is_admin": False,
            "permissions": Role.get_role_permissions()["viewer"]
        }, 
        "user9": {
            "id":2, 
            "space": space_keys[1],
            "is_admin": False,
            "permissions": Role.get_role_permissions()["editor"]
        }
    }
    return users