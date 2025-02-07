import Space
import SpacePermissionEntity

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
            entity = SpacePermissionEntity.SpacePermissionEntity(permission, status, user_permission)
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