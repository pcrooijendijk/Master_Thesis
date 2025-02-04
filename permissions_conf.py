import jsonify

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

class Space: 
    # Space is a collection of documents
    def __init__(self, name, key):
        self.name = name
        self.key = key
        self.permissions = {}
        self.documents = []
    
    def get_permissions(self):
        return self.permissions
    
    def get_space_key(self):
        return self.key
    
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

class SpacePermission:
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

class UserAccessor:
    def __init__(self):
        self.users = {"user1", "user2"}
    
    def get_user(self, username):
        return username if username in self.users else None

class SpaceManager: 

    def __init__(self):
        self.spaces = {}
    
    def get_all_spaces(self):
        return self.spaces
    
    def get_space(self, space_key):
        return self.spaces.get(space_key, None)
    
    def add_space(self, space):
        self.spaces[space.key] = space

class SpacePermissionManager: 

    def __init__(self):
        self.permissions = {}

    def has_permissions(self, permission_type: str, space: Space, username: str):
        return self.permissions.get((space.key, username, permission_type), False)
    
    def save_permission(self, space: Space, username: str, permission_type: str):
        self.permissions[(space.key, username, permission_type)] = True

class SpacePermissionsEntity: 

    def __init__(self, space_name, space_key, permissions, user_permissions):
        self.space_name = space_name
        self.space_key = space_key
        self.permissions = permissions
        self.user_permissions = user_permissions

    def set_space_permissions_status(self, permissions, status, user_permission):
        found = False
        for permission in permissions: 
            if permission.get_permission_type() == permission:
                permission.set_permission_granted(status)
                permission.set_user_permission(user_permission)
                found = True
        
        if not found:
            entity = SpacePermissionsEntity(permission, status, user_permission)
            permissions.append(entity)
    
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

class UserPermissionsEntity:

    def __init__(self, space_permissions):
        self.space_permissions = space_permissions
    
    def get_space_permissions(self):
        return self.get_space_permissions

    def set_space_permissions(self, new_space_permissions):
        self.space_permissions = new_space_permissions

class RestUserPermissionManager:
    def __init__(self, space_manager : SpaceManager, space_permision_manager : SpacePermissionManager, user_accessor : UserAccessor):
        self.space_manager = space_manager
        self.space_permission_manager = space_permision_manager
        self.user_accessor = user_accessor
    
    def get_permission_entity(self, username : str):
        if self.user_accessor.get_user(username) is None: 
            return None
        
        space_permissions = []
        spaces = self.space_manager.get_all_spaces()

        for space in spaces: 
            space_permissions_entity = SpacePermissionsEntity(space.name, space.key)
            self.set_user_space_permission_entity(space_permissions_entity, username, SpacePermission.VIEWSPACE_PERMISSION, space)
            if space_permissions_entity.get_permission_status(SpacePermission.VIEWSPACE_PERMISSION):
                self.set_user_space_permission_entity(space_permissions_entity, username, SpacePermission.REMOVE_OWN_CONTENT_PERMISSION, space)
                self.set_user_space_permission_entity(space_permissions_entity, username, SpacePermission.CREATEEDIT_PAGE_PERMISSION, space)
                self.set_user_space_permission_entity(space_permissions_entity, username, SpacePermission.REMOVE_PAGE_PERMISSION, space)
                self.set_user_space_permission_entity(space_permissions_entity, username, SpacePermission.EDITBLOG_PERMISSION, space)
                self.set_user_space_permission_entity(space_permissions_entity, username, SpacePermission.REMOVE_BLOG_PERMISSION, space)
                self.set_user_space_permission_entity(space_permissions_entity, username, SpacePermission.EDITBLOG_PERMISSION, space)    
                self.set_user_space_permission_entity(space_permissions_entity, username, SpacePermission.REMOVE_BLOG_PERMISSION, space)      
                self.set_user_space_permission_entity(space_permissions_entity, username, SpacePermission.CREATE_ATTACHMENT_PERMISSION, space)    
                self.set_user_space_permission_entity(space_permissions_entity, username, SpacePermission.REMOVE_ATTACHMENT_PERMISSION, space)  
                self.set_user_space_permission_entity(space_permissions_entity, username, SpacePermission.COMMENT_PERMISSION, space)
                self.set_user_space_permission_entity(space_permissions_entity, username, SpacePermission.REMOVE_COMMENT_PERMISSION, space)
                self.set_user_space_permission_entity(space_permissions_entity, username, SpacePermission.SET_PAGE_PERMISSIONS_PERMISSION, space)
                self.set_user_space_permission_entity(space_permissions_entity, username, SpacePermission.REMOVE_MAIL_PERMISSION, space)
                self.set_user_space_permission_entity(space_permissions_entity, username, SpacePermission.EXPORT_SPACE_PERMISSION, space)
                self.set_user_space_permission_entity(space_permissions_entity, username, SpacePermission.ADMINISTER_SPACE_PERMISSION, space)
            
        return UserPermissionsEntity(space_permissions_entity)

    def set_permissions(self, target_user_name : str, user_permissions_entity : UserPermissionsEntity, only_user_permissions : bool):
        space_permissions = user_permissions_entity.get_space_permissions()
        for space_perm in space_permissions:
            granted = False
            view_granted = False

            if space_perm.get_permission_status(SpacePermission.VIEWSPACE_PERMISSION):
                space = self.space_manager.get_space(space.get_space_key())
                view_granted = self.set_space_permission_for_user(space_perm, target_user_name, SpacePermission.VIEWSPACE_PERMISSION, space, only_user_permissions) 
                granted = self.set_space_permission_for_user(space_perm, target_user_name, SpacePermission.REMOVE_OWN_CONTENT_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, SpacePermission.CREATEEDIT_PAGE_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, SpacePermission.REMOVE_PAGE_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, SpacePermission.EDITBLOG_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, SpacePermission.REMOVE_BLOG_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, SpacePermission.CREATE_ATTACHMENT_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, SpacePermission.REMOVE_ATTACHMENT_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, SpacePermission.COMMENT_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, SpacePermission.REMOVE_COMMENT_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, SpacePermission.SET_PAGE_PERMISSIONS_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, SpacePermission.REMOVE_MAIL_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, SpacePermission.EXPORT_SPACE_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, SpacePermission.ADMINISTER_SPACE_PERMISSION, space, only_user_permissions) or granted
            
            if granted and not view_granted:
                if not self.has_user_permission(SpacePermission.VIEWSPACE_PERMISSION, space, target_user_name):
                    space_permission = SpacePermission.create_user_space_permission(SpacePermission.VIEWSPACE_PERMISSION, space, target_user_name)
                    self.space_permission_manager.save_permission(space_permission)


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

    def set_user_space_permission_entity(self, space_permissions_entity : SpacePermissionsEntity, username : str, space_permission_type, space : Space):
        user = self.user_accessor.get_user(username)
        if self.space_permission_manager.has_permission(space_permission_type, space, user):
            user_permission  = False
            space_permissions = space.get_permissions()
            for space_permission in space_permissions:
                if not space_permission.is_user_permission: # Make this a while loop?
                    continue
                elif space_permission.get_user_name() == username: 
                    if space_permission.get_permission_type() == space_permission_type:
                        user_permission = True
                        break
            space_permissions_entity.set_space_permissions_status(space_permission_type, True, user_permission)
        else: 
            space_permissions_entity.set_space_permissions_status(space_permission_type, False, False)
    
    def set_space_permission_for_user(self, space_permissions_entity : SpacePermissionsEntity, username : str, space_permission_type : str, space : Space, only_user_permissions : bool):
        granted = False
        entity = space_permissions_entity.get_space_permission_entity(space_permission_type)

        if entity is not None and entity.is_permission_granted() and not SpacePermissionManager.has_permissions(space_permission_type, space, self.user_accessor.get_user(username)):
            if not only_user_permissions or (only_user_permissions and entity.is_user_permission()):
                space_permission = SpacePermission.create_user_space_permission(space_permission_type, space, username)
                SpacePermissionManager.save_permission(space_permission)
                granted = True
        
        return granted

class UserManager:
    
    def __init__(self):
        self.admins = {"admin"}
        self.users = {"user1", "user2"}

    def get_remote_username(self, req):
        return req.headers.get("Username")
    
    def is_system_admin(self, username):
        return username in self.admins

class UserPermissionsResource:

    def __init__(self, user_manager : UserManager, transaction_template : TransactionTemplate, user_accessor : UserAccessor, space_manager : SpaceManager, space_permission_manager : SpacePermissionManager):
        self.user_manager = user_manager
        self.transaction_template = transaction_template
        self.user_accessor = user_accessor
        self.rest_user_permission_manager = RestUserPermissionManager(space_manager, space_permission_manager, user_accessor)

    def authorize_admin(self, request : str):
        current_username = self.user_manager.get_remote_username(request)
        if not current_username or not self.user_manager.is_system_admin(current_username):
            return jsonify({"error": "Unauthorized"}), 404 


    def get_permissions(self, target_username : str, request):
        current_username = self.user_manager.is_system_admin(current_username)

        if (current_username == None or not self.user_manager.is_system_admin(current_username)):
            return jsonify({"error": "User not found"}), 404 
        
        entity = self.rest_user_permission_manager.get_permission_entity(target_username)
        if entity is None: 
            return jsonify({"error": "User not found"}), 404 
        return entity == None

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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
    # import requests

    # url = "http://127.0.0.1:5000/spaces/space1/documents"
    # headers = {"Content-Type": "application/json"}
    # data = {"doc_name": "doc1"}

    # response = requests.post(url, json=data, headers=headers)
    # print(response.json())
