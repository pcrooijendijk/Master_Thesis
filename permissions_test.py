import json

# Using confluence global permissions we have:
# Different groups represent the different clients which want to access the global model for questions
global_permissions = [
    {
        "permissionType": "Can Use",
        "groups": ["administrators", "space-creators", "confluence-users"]
    },
    {
        "permissionType": "Personal Space",
        "groups": ["administrators", "space-creators", "confluence-users"]
    },
    {
        "permissionType": "Create Space",
        "groups": ["administrators", "space-creators"]
    },
    {
        "permissionType": "Confluence Administrator",
        "groups": ["administrators"]
    },
    {
        "permissionType": "System Administrator",
        "groups": ["administrators"]
    }
    
]

space_permissions = {  
    "spaceKey": "ENG_DEPT",  
    "permissions": [
        {
            "permissionType": "View Content",
            "groups": ["engineering", "management"],
            "users": []
        },
        {
            "permissionType": "Delete Content",
            "groups": ["engineering"],
            "users": ["john.doe"]
        },
        {
            "permissionType": "Delete Pages",
            "groups": ["engineering"],
            "users": ["john.doe"]
        },
        {
            "permissionType": "Add Pages",
            "groups": ["engineering"],
            "users": ["john.doe"]
        },
        {
            "permissionType": "Add Attachment",
            "groups": ["engineering"],
            "users": ["john.doe"]
        },
        {
            "permissionType": "Delete Attachment",
            "groups": ["engineering"],
            "users": ["john.doe"]
        },
        {
            "permissionType": "Add Restrictions",
            "groups": ["engineering"],
            "users": ["john.doe"]
        },
        {
            "permissionType": "Delete Restrictions",
            "groups": ["engineering"],
            "users": ["john.doe"]
        },
        {
            "permissionType": "Delete Mail",
            "groups": ["management"],
            "users": ["john.doe"]
        },
        {
            "permissionType": "Export Space",
            "groups": ["management"],
            "users": ["john.doe"]
        },
        {
            "permissionType": "Admin",
            "groups": ["management"],
            "users": ["john.doe"]
        }
    ]
}

page_restrictions = [
    {
        "pageTitle": "Quarterly Strategy",
        "viewRestrictions": {
            "groups": ["management"],
            "users": ["john.doe"]
        },
        "editRestrictions": {
            "groups": [],
            "users": ["jane.smith"]
        }
    },
    {
        "pageTitle": "Technical Roadmap",
        "viewRestrictions": {
            "groups": ["engineering", "management"],
            "users": []
        },
        "editRestrictions": {
            "groups": ["engineering"],
            "users": []
        }
    }
] 

permissions_structure = {
    "globalPermissions": global_permissions,
    "spacePermissions": space_permissions,
    "pageRestrictions": page_restrictions
}

print("test?")

# Convert the permissions structure to a JSON string
permissions_json = json.dumps(permissions_structure, indent=4)

documents = [
    {
        "title": "blabla",
        "content": "blabla", 
        "spaceKey": "ENG"
    }, 
    {
        "title": "blabla",
        "content": "blabla", 
        "spaceKey": "DEV"
    }
]