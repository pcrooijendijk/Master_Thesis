#%%
import jsonify
from enum import Enum

# Permission Entity
class SpacePermissionEntity:

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

# Space is a collection of documents
class Space: 
    
    def __init__(self, name, key):
        self.name = name
        self.key = key
        self.permissions = {}
        self.documents = []
    
    def get_permissions(self):
        return self.permissions
    
    def get_space_key(self):
        return self.key
        
    def get_name(self):
        return self.name
    
    def add_document(self, document):
        self.documents.append(document)

class TransactionTemplate: 

    def execute(self, callback):
        try: 
            result = callback()
            return result
        except Exception as e: 
            print(f"Transaction failed: {e}")
            return None

# The different permissions for a space
class Permission(Enum): 
    VIEWSPACE_PERMISSION = "VIEWSPACE_PERMISSION"
    REMOVE_OWN_CONTENT_PERMISSION = "REMOVE_OWN_CONTENT_PERMISSION"
    CREATEEDIT_PAGE_PERMISSION = "CREATEEDIT_PAGE_PERMISSION"
    REMOVE_PAGE_PERMISSION = "REMOVE_PAGE_PERMISSION"
    EDITBLOG_PERMISSION = "EDITBLOG_PERMISSION"
    REMOVE_BLOG_PERMISSION = "REMOVE_BLOG_PERMISSION"
    CREATE_ATTACHMENT_PERMISSION = "CREATE_ATTACHMENT_PERMISSION"
    REMOVE_ATTACHMENT_PERMISSION = "REMOVE_ATTACHMENT_PERMISSION"
    COMMENT_PERMISSION = "COMMENT_PERMISSION"
    REMOVE_COMMENT_PERMISSION = "REMOVE_COMMENT_PERMISSION"
    SET_PAGE_PERMISSIONS_PERMISSION = "SET_PAGE_PERMISSIONS_PERMISSION"
    REMOVE_MAIL_PERMISSION = "REMOVE_MAIL_PERMISSION"
    EXPORT_SPACE_PERMISSION = "EXPORT_SPACE_PERMISSION"
    ADMINISTER_SPACE_PERMISSION = "ADMINISTER_SPACE_PERMISSION"

# Space Permission: permissions of a user on a space
class SpacePermission:

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

# User Accessor keeps track of users
class UserAccessor:
    def __init__(self):
        self.users = {"user1", "user2", "admin"}
    
    def get_user(self, username):
        return username if username in self.users else None

# Space Manager keeps track of spaces
class SpaceManager: 

    def __init__(self):
        self.spaces = {}
    
    def get_all_spaces(self):
        return self.spaces
    
    def get_space(self, space_key):
        return self.spaces.get(space_key, None)
    
    def add_space(self, space):
        self.spaces[space.key] = space

# Space Permission Manager keeps track of space permissions
class SpacePermissionManager: 

    def __init__(self):
        self.permissions = {}
    
    # Check if a user has a permission on a space
    def has_permissions(self, space_permission_type: str, space: Space, username: str):
        if space.name in self.permissions and username in self.permissions[space.name]:
            return self.permissions[space.name][username]
        else:
            return None
    
    # Save permission on a space for a user
    def save_permission(self, space: Space, username: str, permission_type: str):
        if space.name not in self.permissions:
            self.permissions[space.name] = {}
        self.permissions[space.name][username] = permission_type

# Space Permissions Entity keeps track of space permissions
class SpacePermissionsEntity: 

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

# User Permissions Entity keeps track of user permissions
class UserPermissionsEntity:

    def __init__(self, space_permissions):
        self.space_permissions = space_permissions
    
    def get_space_permissions(self):
        return self.space_permissions

    def set_space_permissions(self, new_space_permissions):
        self.space_permissions = new_space_permissions

# Rest User Permission Manager keeps track of user permissions per space
class RestUserPermissionManager:
    def __init__(self, space_manager : SpaceManager, space_permission_manager : SpacePermissionManager, user_accessor : UserAccessor):
        self.space_manager = space_manager
        self.space_permission_manager = space_permission_manager
        self.user_accessor = user_accessor
    
    # Get permission entity for a user
    def get_permission_entity(self, username: str):
        entity = None

        # Only continue if the user exists
        if self.user_accessor.get_user(username) is not None: 
            space_permissions = []
            spaces = self.space_manager.get_all_spaces()

            for space in spaces: 
                space_permissions_entity = SpacePermissionsEntity(spaces[space].name, spaces[space].key, spaces[space])
                self.set_user_space_permission_entity(space_permissions_entity, username, Permission.VIEWSPACE_PERMISSION, spaces[space])
                if space_permissions_entity.get_permission_status(Permission.VIEWSPACE_PERMISSION):
                    self.set_user_space_permission_entity(space_permissions_entity, username, Permission.REMOVE_OWN_CONTENT_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, Permission.CREATEEDIT_PAGE_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, Permission.REMOVE_PAGE_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, Permission.EDITBLOG_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, Permission.REMOVE_BLOG_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, Permission.EDITBLOG_PERMISSION, spaces[space])    
                    self.set_user_space_permission_entity(space_permissions_entity, username, Permission.REMOVE_BLOG_PERMISSION, spaces[space])      
                    self.set_user_space_permission_entity(space_permissions_entity, username, Permission.CREATE_ATTACHMENT_PERMISSION, spaces[space])    
                    self.set_user_space_permission_entity(space_permissions_entity, username, Permission.REMOVE_ATTACHMENT_PERMISSION, spaces[space])  
                    self.set_user_space_permission_entity(space_permissions_entity, username, Permission.COMMENT_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, Permission.REMOVE_COMMENT_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, Permission.SET_PAGE_PERMISSIONS_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, Permission.REMOVE_MAIL_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, Permission.EXPORT_SPACE_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, Permission.ADMINISTER_SPACE_PERMISSION, spaces[space])
                    space_permissions.append(space_permissions_entity)
            entity = UserPermissionsEntity(space_permissions)
        return entity

    # Set permissions for a user
    def set_permissions(self, target_user_name : str, user_permissions_entity : UserPermissionsEntity, only_user_permissions : bool):
        space_permissions = user_permissions_entity.get_space_permissions()
        for space_perm in space_permissions:
            granted = False
            view_granted = False

            if space_perm.get_permission_status(Permission.VIEWSPACE_PERMISSION):
                space = self.space_manager.get_space(space.get_space_key())
                view_granted = self.set_space_permission_for_user(space_perm, target_user_name, Permission.VIEWSPACE_PERMISSION, space, only_user_permissions) 
                granted = self.set_space_permission_for_user(space_perm, target_user_name, Permission.REMOVE_OWN_CONTENT_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, Permission.CREATEEDIT_PAGE_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, Permission.REMOVE_PAGE_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, Permission.EDITBLOG_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, Permission.REMOVE_BLOG_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, Permission.CREATE_ATTACHMENT_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, Permission.REMOVE_ATTACHMENT_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, Permission.COMMENT_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, Permission.REMOVE_COMMENT_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, Permission.SET_PAGE_PERMISSIONS_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, Permission.REMOVE_MAIL_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, Permission.EXPORT_SPACE_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, Permission.ADMINISTER_SPACE_PERMISSION, space, only_user_permissions) or granted
            
            if granted and not view_granted:
                if not self.has_user_permission(Permission.VIEWSPACE_PERMISSION, space, target_user_name):
                    space_permission = SpacePermission.create_user_space_permission(Permission.VIEWSPACE_PERMISSION, space, target_user_name)
                    self.space_permission_manager.save_permission(space_permission)


    # Check if user has permission
    def has_user_permission(self, space_permission_type : str, space : Space, username : str):
        user_permission = False

        if self.space_permission_manager.has_permissions(space_permission_type, space, self.user_accessor.get_user(username)):
            space_permissions = space.get_permissions()
            for space_permission in space_permissions:
                if not space_permission.is_user_permission():
                    continue
                elif space_permission.get_user_name() == username: 
                    if space_permission.get_type() == space_permission_type:
                        user_permission = True
                        break
        
        return user_permission

    # Set user space permission
    def set_user_space_permission_entity(self, space_permissions_entity : SpacePermissionsEntity, username : str, space_permission_type, space : Space):
        user = self.user_accessor.get_user(username)
        if self.space_permission_manager.has_permissions(space_permission_type, space, user):
            user_permission  = False
            space_permissions = space.get_permissions()
            for space_permission in space_permissions:
                if not space_permission.is_user_permission: # Make this a while loop?
                    continue
                elif space_permission.get_user_name() == username: 
                    if space_permission.get_permission_type() == space_permission_type:
                        user_permission = True
                        break
            space_permissions_entity.set_space_permissions_status(space_permission_type, True, user_permission, username)
        else: 
            space_permissions_entity.set_space_permissions_status(space_permission_type, False, False, username)
    
    # Set space permission
    def set_space_permission_for_user(self, space_permissions_entity : SpacePermissionsEntity, username : str, space_permission_type : str, space : Space, only_user_permissions : bool):
        granted = False
        entity = space_permissions_entity.get_space_permission_entity(space_permission_type)

        if entity is not None and entity.is_permission_granted() and not SpacePermissionManager.has_permissions(space_permission_type, space, self.user_accessor.get_user(username)):
            if not only_user_permissions or (only_user_permissions and entity.is_user_permission()):
                space_permission = SpacePermission.create_user_space_permission(space_permission_type, space, username)
                SpacePermissionManager.save_permission(space_permission)
                granted = True
        
        return granted

# User Manager keeps track of the users in the system
class UserManager:
    
    def __init__(self):
        self.admins = {"admin", "user1"}
        self.users = {"user1", "user2"}

    def get_remote_username(self, req):
        return req.get("Username")
    
    def is_system_admin(self, username):
        return username in self.admins

# User Permissions Resource is used to get and set user permissions
class UserPermissionsResource:

    def __init__(self, user_manager : UserManager, transaction_template : TransactionTemplate, user_accessor : UserAccessor, space_manager : SpaceManager, space_permission_manager : SpacePermissionManager):
        self.user_manager = user_manager
        self.transaction_template = transaction_template
        self.user_accessor = user_accessor
        self.rest_user_permission_manager = RestUserPermissionManager(space_manager, space_permission_manager, user_accessor)

    def authorize_admin(self, request: str):
        current_username = self.user_manager.get_remote_username(request)
        if not current_username or not self.user_manager.is_system_admin(current_username):
            return jsonify({"error": "Unauthorized"}), 404 

    def get_permissions(self, target_username: str, request):
        current_username = self.user_manager.get_remote_username(request)

        if (current_username is None or not self.user_manager.is_system_admin(current_username)):
            return "error: User not found"
        
        entity = self.rest_user_permission_manager.get_permission_entity(target_username)
        if entity is None: 
            return "error: User not found"

        space_permissions = []

        for space in entity.get_space_permissions():
            space_data = {
            "spaceName": space.get_space_name(),
            "spaceKey": space.get_space_key(),
            "permissions": [
                {
                    "permissionType": perm.get_permission_type(),
                    "permissionGranted": perm.is_permission_granted(),
                    "userPermission": perm.is_user_permission()
                } for perm in space.get_permissions()]
            }
            space_permissions.append(space_data)
        return space_permissions

    def put(self, target_username : str, only_user_permissions: bool, user_permissions_entity : UserPermissionsEntity, request):
        username = self.user_manager.get_remote_username(request)
        if username == None or not self.user_manager.is_system_admin(username):
            return jsonify({"error": "Unauthorized"}), 404 
        
        if self.user_accessor.get_user(target_username) is None: 
            return jsonify({"error": "Not found"}), 404 
        
        def transaction():
            self.rest_user_permission_manager.set_permissions(username, user_permissions_entity, only_user_permissions)
        
        self.transaction_template.execute(transaction)

from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize managers
space_manager = {}
space_permission_manager = SpacePermissionManager()
transaction_template = TransactionTemplate()

@app.route("/spaces")
def create_space():
    """Create a new space (document collection)."""
    data = request.json
    space_key = data["key"]
    space_manager[space_key] = Space(data["name"], space_key)
    return jsonify({"message": f"Space '{data['name']}' created."}), 201

@app.route("/spaces/<space_key>/documents")
def add_document(space_key):
    """Add a document to a space."""
    if space_key not in space_manager:
        return jsonify({"error": "Space not found"}), 404
    
    data = request.json
    space_manager[space_key].add_document(data["doc_name"])
    return jsonify({"message": f"Document '{data['doc_name']}' added to space '{space_key}'"}), 201

@app.route("/spaces/<space_key>/permissions/<username>")
def get_permissions(space_key, username):
    """Get user permissions for a space."""
    permissions = [perm.permission_type for perm in space_permission_manager.permissions if perm.username == username]
    return jsonify({"username": username, "permissions": permissions})

@app.route("/spaces/<space_key>/permissions/<username>")
def update_permissions(space_key, username):
    """Grant a user new permissions."""
    data = request.json
    permission_type = data["permission"]

    def transaction():
        space_permission_manager.grant_permission(space_key, username, permission_type)

    transaction_template.execute(transaction)
    return jsonify({"message": f"Permission '{permission_type}' granted to '{username}'"}), 200

@app.route("/")
def home():
    return "Flask is working!", 200

#%%
import os
import pymupdf
import random

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
space = Space('mark', 0)
space_new = Space('new', 1)

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
space_permission_manager.save_permission(space, 'admin', Permission.VIEWSPACE_PERMISSION)
space_permission_manager.save_permission(space_new, 'user1', Permission.REMOVE_OWN_CONTENT_PERMISSION)

# Getting the permissions for user with admin permissions
user_permissions_resource = UserPermissionsResource(UserManager(), TransactionTemplate(), UserAccessor(), space_manager, space_permission_manager)

# Get the permissions for user admin (target username, request)
print(user_permissions_resource.get_permissions('admin', {"Username": "admin"}))
print("----------------------------------------------------------")
print(user_permissions_resource.get_permissions('user1', {"Username": "user1"}))
