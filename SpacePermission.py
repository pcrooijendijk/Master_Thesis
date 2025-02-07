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