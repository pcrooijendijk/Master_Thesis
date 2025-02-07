class UserPermissionsEntity:
    """
        User Permissions Entity keeps track of user permissions
    """

    def __init__(self, space_permissions):
        self.space_permissions = space_permissions
    
    def get_space_permissions(self):
        return self.space_permissions

    def set_space_permissions(self, new_space_permissions):
        self.space_permissions = new_space_permissions