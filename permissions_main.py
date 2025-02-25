import pymupdf
import random
from pathlib import Path

import Space
import utils.Permission as Permission
import TransactionTemplate
from fed_utils.Client import Client
from UserPermissions import UserAccessor
from SpacePermissions import SpaceManager
from SpacePermissions import SpacePermissionManager
from UserPermissions import UserManager
from UserPermissionManagement import UserPermissionsResource


file_path = Path("/media/sf_thesis/data_DocBench_test")
texts = []

# Append the documents to texts lists to get the content of the documents
for pdf in file_path.rglob("*.pdf"): # Use rglob to find all PDFs
    with pymupdf.open(pdf) as file: 
        space_key_index = random.randint(0, 3)  # For the space key
        texts.append(
            [
                chr(12).join([page.get_text() for page in file]),
                file.metadata,
                space_key_index,
            ]
        )

# Template for role permissions (not mandatory to use these)
role_permissions = {
    "admin": list(Permission.Permission),  # Full access with all permissions
    "editor": [Permission.Permission.VIEWSPACE_PERMISSION, Permission.Permission.CREATEEDIT_PAGE_PERMISSION, Permission.Permission.COMMENT_PERMISSION],
    "viewer": [Permission.Permission.VIEWSPACE_PERMISSION],
    "restricted_user": []  # No permissions
}

# Making a space, space keys are defined (for now) as integers
space = Space.Space('mark', 0)
space_new = Space.Space('new', 1)
space_3 = Space.Space("lol", 2)

# Adding two documents to the space
space.add_document(texts[0]) 
space.add_document(texts[1])
space_new.add_document(texts[2])
space_new.add_document(texts[3])
space_3.add_document(texts[2])

# Make a space manager
space_manager = SpaceManager()
# Adding the spaces
space_manager.add_space(space) 
space_manager.add_space(space_new) 
space_manager.add_space(space_3)

# Adding users
user_accessor = UserAccessor()
user_accessor.add_user("admin")
user_accessor.add_user("user1")

# Adding the admin
user_manager = UserManager(user_accessor)
user_manager.add_admin("admin")

# Make a space permission manager
space_permission_manager = SpacePermissionManager()

# Adding permissions for admin (space, username, permission type)
space_permission_manager.save_permission(space, 'admin', role_permissions["editor"])
space_permission_manager.save_permission(space_new, 'user1', role_permissions["viewer"])
space_permission_manager.save_permission(space_3, 'admin', role_permissions["editor"])

# Getting the permissions for user with admin permissions
user_permissions_resource = UserPermissionsResource(user_manager, TransactionTemplate.TransactionTemplate(), user_accessor, space_manager, space_permission_manager)

# Get the permissions for user admin (target username, request)
# print(user_permissions_resource.get_permissions('admin', {"Username": "admin"}))
# print("----------------------------------------------------------")
# print(user_permissions_resource.get_permissions('user1', {"Username": "user1"}))

# Make the clients by using the Client class:
admin_client = Client(1, "admin", user_permissions_resource, "DeepSeek-v3:1.5b")
admin_client.get_spaces()
admin_client.filter_documents()