from enum import Enum
from perm_utils import Permission

class Role(Enum):
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"
    RESTRICTED_USER = "restricted_user"

    def get_role_permissions() -> dict:
        # Template for role permissions (not mandatory to use these)
        role_permissions = {
            "admin": list(Permission),  # Full access with all permissions
            "editor": [Permission.VIEWSPACE_PERMISSION, Permission.CREATEEDIT_PAGE_PERMISSION, Permission.COMMENT_PERMISSION],
            "viewer": [Permission.VIEWSPACE_PERMISSION],
            "restricted_user": []  # No permissions
        }

        return role_permissions