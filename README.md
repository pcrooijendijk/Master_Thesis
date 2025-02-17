# Master_Thesis
Repository for my Master Thesis

Overview of the files: 
- `Permission.py` contains the different permissions for a space saved as an enum.
- `permission_*.py` are the 'main' files where the structure is tested.
- `Space.py` is the structure which defines the space where each space has its own name, key, set of permissions and documents.
- `SpacePermissions.py` has all the relevant space permissions classes: SpaceManager, SpacePermission, SpacePermissionEntity, SpacePermissionManager and SpacePermissionsEntity.
- `TransactionTemplate.py` is still under construction and is currently not (actively) used.
- `UserPermissionManagement.py` contains the RestUserPermissionManager and UserPermissionsResource which both handle the permissions and users correctly.
- `UserPermissions.py` has all the relevant user permissions classes: UserAccessor, UserManager and UserPermissionsEntity. 