import perm_utils.Space as Space
from perm_utils import Permission
from typing import List

class SpaceManager: 
    """
        Space Manager keeps track of spaces
    """

    def __init__(self):
        self.spaces = {}
    
    def get_all_spaces(self) -> dict:
        return self.spaces
    
    def get_space(self, space_key: str) -> dict:
        return self.spaces.get(space_key, None)
    
    def add_space(self, space: Space) -> None:
        self.spaces[space.key] = space

class SpacePermission:
    """
        Space Permission: permissions of a user on a space
    """

    def __init__(self, space: Space, username: str, permission_type: Permission, is_user_permission: bool = True):
        self.space = space
        self.username = username
        self.permission_type = permission_type
        self.is_user_permission = is_user_permission

    def get_user_name(self) -> str:
        return self.username
    
    def get_permission_type(self) -> Permission:
        return self.permission_type

class SpacePermissionEntity:
    """
    Permission Entity
    """

    def __init__(self, permission_type: Permission, permission_granted: bool, user_permission: bool):
        self.permission_type = permission_type
        self.permission_granted = permission_granted
        self.user_permission = user_permission

    def is_user_permission(self) -> bool:
        return self.user_permission
    
    def set_user_permission(self, new_user_permission) -> None:
        self.user_permission = new_user_permission

    def get_permission_type(self) -> Permission:
        return self.permission_type
    
    def set_permission_type(self, new_permission_type: Permission) -> None:
        self.permission_type = new_permission_type
    
    def is_permission_granted(self) -> bool:
        return self.permission_granted
    
    def set_permission_granted(self, new_permission_granted: bool) -> None:
        self.permission_granted = new_permission_granted

class SpacePermissionManager: 
    """
        Space Permission Manager keeps track of space permissions
    """
    def __init__(self):
        self.permissions = {}
    
    # Check if a user has a permission on a space
    def has_permissions(self, space_permission_type: str, space: Space, username: str) -> bool:
        if space.name in self.permissions and username in self.permissions[space.name]:
            return self.permissions[space.name][username]
        else:
            return None
    
    # Save permission on a space for a user
    def save_permission(self, space: Space, username: str, permission_type: List) -> None:
        if space.name not in self.permissions:
            self.permissions[space.name] = {}
        self.permissions[space.name][username] = [perm.value for perm in permission_type]
    
    def get_permissions(self) -> dict:
        return self.permissions

class SpacePermissionsEntity: 
    """
        Space Permissions Entity keeps track of space permissions
    """

    def __init__(self, space_name: str, space_key: str, space: Space):
        self.space_name = space_name
        self.space_key = space_key
        self.space = space
        self.permissions = [] # Should be a list of SpacePermissionEntity

    # Set a permission for a user
    def set_space_permissions_status(self, permission: Permission, status: bool, user_permission: bool) -> None:
        found = False # Flag to check if the permission was found
        for perm in self.permissions: 
            if perm.get_permission_type() == permission: # Check if the permission is of the correct type
                perm.set_permission_granted(status)
                perm.set_user_permission(user_permission)
                found = True
        
        if not found:
            # Adding the permission by making a new entity
            entity = SpacePermissionEntity(permission, status, user_permission)
            self.permissions.append(entity)
    
    def get_permission_status(self, permission: Permission) -> bool:
        status = False
        for perm in self.permissions:
            if perm.get_permission_type() == permission:
                status = perm.is_permission_granted()
        
        return status

    def get_space_permission_entity(self, permission: Permission) -> SpacePermissionEntity:
        entity = None
        for perm in self.permissions:
            if perm.get_permission_type() == permission:
                entity = perm
        
        return entity
    
    def get_space_name(self) -> str:
        return self.space_name

    def set_space_name(self, new_space_name: str) -> None:
        self.space_name = new_space_name
    
    def get_space_key(self) -> str:
        return self.space_key
    
    def set_space_key(self, new_space_key: str) -> None:
        self.space_key = new_space_key

    def get_permissions(self) -> Permission: 
        return self.permissions
    
    def set_permissions(self, new_permissions: Permission) -> None:
        self.permissions = new_permissions