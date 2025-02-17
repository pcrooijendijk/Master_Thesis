import Space

class SpaceManager: 
    """
        Space Manager keeps track of spaces
    """

    def __init__(self):
        self.spaces = {}
    
    def get_all_spaces(self):
        return self.spaces
    
    def get_space(self, space_key):
        return self.spaces.get(space_key, None)
    
    def add_space(self, space):
        self.spaces[space.key] = space

class SpacePermission:
    """
        Space Permission: permissions of a user on a space
    """

    def __init__(self, space, username, permission_type, is_user_permission=True):
        self.space = space
        self.username = username
        self.permission_type = permission_type
        self.is_user_permission = is_user_permission

    def create_user_space_permission(permission_type, space, username):
        return SpacePermission(space, username, permission_type, is_user_permission=True)

    def get_user_name(self):
        return self.username
    
    def get_permission_type(self):
        return self.permission_type

class SpacePermissionEntity:
    """
    Permission Entity
    """

    def __init__(self, permission_type, permission_granted, user_permission):
        self.permission_type = permission_type
        self.permission_granted = permission_granted
        self.user_permission = user_permission

    def is_user_permission(self):
        return self.user_permission
    
    def set_user_permission(self, new_user_permission):
        self.user_permission = new_user_permission

    def get_permission_type(self):
        return self.permission_type
    
    def set_permission_type(self, new_permission_type):
        self.permission_type = new_permission_type
    
    def is_permission_granted(self):
        return self.permission_granted
    
    def set_permission_granted(self, new_permission_granted):
        self.permission_granted = new_permission_granted

import Space

class SpacePermissionManager: 
    """
        Space Permission Manager keeps track of space permissions
    """
    def __init__(self):
        self.permissions = {}
    
    # Check if a user has a permission on a space
    def has_permissions(self, space_permission_type: str, space: Space, username: str):
        if space.name in self.permissions and username in self.permissions[space.name]:
            return self.permissions[space.name][username]
        else:
            return None
    
    # Save permission on a space for a user
    def save_permission(self, space: Space, username: str, permission_type: list):
        if space.name not in self.permissions:
            self.permissions[space.name] = {}
        self.permissions[space.name][username] = [perm.value for perm in permission_type]
    
    def get_permissions(self):
        return self.permissions

class SpacePermissionsEntity: 
    """
        Space Permissions Entity keeps track of space permissions
    """

    def __init__(self, space_name, space_key, space : Space):
        self.space_name = space_name
        self.space_key = space_key
        self.permissions = [] # Should be a list of SpacePermissionEntity

    # Set a permission for a user
    def set_space_permissions_status(self, permission, status, user_permission, username):
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
    
    def get_permission_status(self, permission):
        status = False
        for perm in self.permissions:
            if perm.get_permission_type() == permission:
                status = perm.is_permission_granted()
        
        return status

    def get_space_permission_entity(self, permission):
        entity = None
        for perm in self.permissions:
            if perm.get_permission_type() == permission:
                entity = perm
        
        return entity
    
    def get_space_name(self):
        return self.space_name

    def set_space_name(self, new_space_name):
        self.space_name = new_space_name
    
    def get_space_key(self):
        return self.space_key
    
    def set_space_key(self, new_space_key):
        self.space_key = new_space_key

    def get_permissions(self): 
        return self.permissions
    
    def set_permissions(self, new_permissions):
        self.permissions = new_permissions