#%%
import os
import pymupdf
import random

import UserAccessor
import Space
import TransactionTemplate
from SpaceManager import *
from SpacePermissionManager import *
import Permission
import UserManager
import UserPermissionsResource


file_path = "/media/sf_thesis/data_DocBench_test"
texts = []

# Append the documents to texts lists to get the content of the documents
for x in os.listdir(file_path):
    for y in os.listdir(file_path + "/" + x):
        if y.endswith(".pdf"):
            with pymupdf.open(file_path + "/" + x + "/" + y) as doc:
                space_key_index = random.randint(0, 3)  # For the space key
                texts.append(
                    [
                        chr(12).join([page.get_text() for page in doc]),
                        doc.metadata,
                        space_key_index,
                    ]
                )

#%%
# Making a space, space keys are defined (for now) as integers
space = Space.Space('mark', 0)
space_new = Space.Space('new', 1)

# Adding two documents to the space
space.add_document(texts[0]) 
space.add_document(texts[1])
space_new.add_document(texts[2])
space_new.add_document(texts[3])

# Make a space manager
space_manager = SpaceManager()
# Adding the spaces
space_manager.add_space(space) 
space_manager.add_space(space_new) 

# Make a space permission manager
space_permission_manager = SpacePermissionManager()

# Adding permissions for admin (space, username, permission type)
space_permission_manager.save_permission(space, 'admin', Permission.Permission.VIEWSPACE_PERMISSION)
space_permission_manager.save_permission(space_new, 'user1', Permission.Permission.REMOVE_OWN_CONTENT_PERMISSION)

# Getting the permissions for user with admin permissions
user_permissions_resource = UserPermissionsResource.UserPermissionsResource(UserManager.UserManager(), TransactionTemplate.TransactionTemplate(), UserAccessor.UserAccessor(), space_manager, space_permission_manager)

# Get the permissions for user admin (target username, request)
print(user_permissions_resource.get_permissions('admin', {"Username": "admin"}))
print("----------------------------------------------------------")
print(user_permissions_resource.get_permissions('user1', {"Username": "user1"}))
