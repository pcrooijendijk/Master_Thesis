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