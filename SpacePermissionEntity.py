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