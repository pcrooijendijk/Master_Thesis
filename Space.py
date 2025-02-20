class Space: 
    """
        Space is a collection of documents
    """
    
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
    
    def get_documents(self):
        return self.documents