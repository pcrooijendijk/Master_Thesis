import perm_utils.Permission as Permission
import perm_utils.Space as Space
from perm_utils.TransactionTemplate import TransactionTemplate
from typing import List

from perm_utils.SpacePermissions import SpaceManager, SpacePermissionManager
from perm_utils.UserPermissions import UserAccessor, UserManager
from perm_utils.UserPermissionManagement import UserPermissionsResource

class SpaceManagement:
    def __init__(self, space_names: List, space_keys: List, documents: List, users: dict):
        self.space_names = space_names
        self.space_keys = space_keys
        self.documents = documents
        self.users = list(users.keys())
        self.user_list = users
        self.admin_list = [user for user, is_admin in users.items() if is_admin["is_admin"]]

        # Initialize all the managers and accessors
        self.space_manager = SpaceManager()
        self.make_spaces()
        self.user_accessor = UserAccessor()
        self.adding_users()
        self.user_manager = UserManager(self.user_accessor)
        self.adding_admin()
        self.space_permission_manager = SpacePermissionManager()
        self.save_permissions()
        self.user_permissions_resource = UserPermissionsResource(
            self.user_manager, 
            TransactionTemplate(), 
            self.user_accessor,
            self.space_manager,
            self.space_permission_manager
            )

    def make_spaces(self) -> None:
        assert len(self.space_names) is len(self.space_keys), "Every space name needs a space key!"

        for index in range(len(self.space_names)):
            # Make the space by using the space name and the space key
            space = Space(self.space_names[index], self.space_keys[index])
            # Adding the documents with the correct space key
            space = self.add_document(space)
            # Adding the space to the space manager
            self.space_manager.add_space(space) 

    # Helper function for make_spaces
    def add_document(self, space: Space) -> Space:
        for index, document in enumerate(self.documents['train']):
            space_key_index = document["space_key_index"]
            if space.get_space_key() is space_key_index:
                space.add_document(self.documents["train"][index])
        return space
    
    def adding_users(self) -> None:
        list(map(self.user_accessor.add_user, self.users))
    
    def adding_admin(self) -> None:
        list(map(self.user_manager.add_admin, self.admin_list))
    
    def save_permissions(self) -> None:
        for user in self.users:
            space = self.space_manager.get_space(self.user_list[user]["space"])
            permissions = self.user_list[user]["permissions"]
            self.space_permission_manager.save_permission(space, user, permissions)
    
    def get_user_permissions_resource(self) -> UserPermissionsResource:
        return self.user_permissions_resource