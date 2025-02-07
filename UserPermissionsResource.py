import UserManager
import TransactionTemplate
import UserAccessor
import RestUserPermissionManager
import UserPermissionsEntity
import SpaceManager
import SpacePermissionManager

import jsonify

class UserPermissionsResource:
    """
        User Permissions Resource is used to get and set user permissions
    """

    def __init__(self, user_manager : UserManager, transaction_template : TransactionTemplate, user_accessor : UserAccessor, space_manager : SpaceManager, space_permission_manager : SpacePermissionManager):
        self.user_manager = user_manager
        self.transaction_template = transaction_template
        self.user_accessor = user_accessor
        self.rest_user_permission_manager = RestUserPermissionManager.RestUserPermissionManager(space_manager, space_permission_manager, user_accessor)

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
