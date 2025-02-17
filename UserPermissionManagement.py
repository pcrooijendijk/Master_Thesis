import Space
import Permission as per
import TransactionTemplate
from SpacePermissions import SpaceManager
from SpacePermissions import SpacePermissionManager
from UserPermissions import UserAccessor
from SpacePermissions import SpacePermissionsEntity
from UserPermissions import UserPermissionsEntity
from SpacePermissions import SpacePermission
from UserPermissions import UserManager

import jsonify

class RestUserPermissionManager:
    """
        Rest User Permission Manager keeps track of user permissions per space
    """

    def __init__(self, space_manager : SpaceManager, space_permission_manager : SpacePermissionManager, user_accessor : UserAccessor):
        self.space_manager = space_manager
        self.space_permission_manager = space_permission_manager
        self.user_accessor = user_accessor

    def get_space_permission_manager(self):
        return self.space_permission_manager
    
    # Get permission entity for a user
    def get_permission_entity(self, username: str):
        entity = None

        # Only continue if the user exists
        if self.user_accessor.get_user(username) is not None: 
            space_permissions = []
            spaces = self.space_manager.get_all_spaces()

            for space in spaces: 
                space_permissions_entity = SpacePermissionsEntity(spaces[space].name, spaces[space].key, spaces[space])
                self.set_user_space_permission_entity(space_permissions_entity, username, per.Permission.VIEWSPACE_PERMISSION, spaces[space])
                if space_permissions_entity.get_permission_status(per.Permission.VIEWSPACE_PERMISSION):
                    self.set_user_space_permission_entity(space_permissions_entity, username, per.Permission.REMOVE_OWN_CONTENT_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, per.Permission.CREATEEDIT_PAGE_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, per.Permission.REMOVE_PAGE_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, per.Permission.EDITBLOG_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, per.Permission.REMOVE_BLOG_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, per.Permission.EDITBLOG_PERMISSION, spaces[space])    
                    self.set_user_space_permission_entity(space_permissions_entity, username, per.Permission.REMOVE_BLOG_PERMISSION, spaces[space])      
                    self.set_user_space_permission_entity(space_permissions_entity, username, per.Permission.CREATE_ATTACHMENT_PERMISSION, spaces[space])    
                    self.set_user_space_permission_entity(space_permissions_entity, username, per.Permission.REMOVE_ATTACHMENT_PERMISSION, spaces[space])  
                    self.set_user_space_permission_entity(space_permissions_entity, username, per.Permission.COMMENT_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, per.Permission.REMOVE_COMMENT_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, per.Permission.SET_PAGE_PERMISSIONS_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, per.Permission.REMOVE_MAIL_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, per.Permission.EXPORT_SPACE_PERMISSION, spaces[space])
                    self.set_user_space_permission_entity(space_permissions_entity, username, per.Permission.ADMINISTER_SPACE_PERMISSION, spaces[space])
                    space_permissions.append(space_permissions_entity)
            entity = UserPermissionsEntity(space_permissions)
        return entity

    # Set permissions for a user
    def set_permissions(self, target_user_name : str, user_permissions_entity : UserPermissionsEntity, only_user_permissions : bool):
        space_permissions = user_permissions_entity.get_space_permissions()
        for space_perm in space_permissions:
            granted = False
            view_granted = False

            if space_perm.get_permission_status(per.Permission.VIEWSPACE_PERMISSION):
                space = self.space_manager.get_space(space.get_space_key())
                view_granted = self.set_space_permission_for_user(space_perm, target_user_name, per.Permission.VIEWSPACE_PERMISSION, space, only_user_permissions) 
                granted = self.set_space_permission_for_user(space_perm, target_user_name, per.Permission.REMOVE_OWN_CONTENT_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, per.Permission.CREATEEDIT_PAGE_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, per.Permission.REMOVE_PAGE_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, per.Permission.EDITBLOG_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, per.Permission.REMOVE_BLOG_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, per.Permission.CREATE_ATTACHMENT_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, per.Permission.REMOVE_ATTACHMENT_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, per.Permission.COMMENT_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, per.Permission.REMOVE_COMMENT_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, per.Permission.SET_PAGE_PERMISSIONS_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, per.Permission.REMOVE_MAIL_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, per.Permission.EXPORT_SPACE_PERMISSION, space, only_user_permissions) or granted
                granted = self.set_space_permission_for_user(space_perm, target_user_name, per.Permission.ADMINISTER_SPACE_PERMISSION, space, only_user_permissions) or granted
            
            if granted and not view_granted:
                if not self.has_user_permission(per.Permission.VIEWSPACE_PERMISSION, space, target_user_name):
                    space_permission = SpacePermission.create_user_space_permission(per.Permission.VIEWSPACE_PERMISSION, space, target_user_name)
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

class UserPermissionsResource:
    """
        User Permissions Resource is used to get and set user permissions
    """

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

        if (current_username is None):
            return "error: User not found"
        
        entity = self.rest_user_permission_manager.get_permission_entity(target_username)
        space_permission_manager = self.rest_user_permission_manager.get_space_permission_manager()
        
        if entity is None: 
            return "error: User not found"

        space_permissions = []

        for space in entity.get_space_permissions():
            # Get the permissions for the current user
            permissions = space_permission_manager.get_permissions()[space.get_space_name()][current_username]

            space_data = {
            "spaceName": space.get_space_name(),
            "spaceKey": space.get_space_key(),
            "permissions": [
                {
                    "permissionType": perm.get_permission_type().value,
                    "permissionGranted": perm.get_permission_type().value in permissions,
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
