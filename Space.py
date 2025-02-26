class Space: 
    """
        Space is a collection of documents
    """
    
    def __init__(self, name: str, key: str):
        self.name = name
        self.key = key
        self.permissions = {}
        self.documents = []
    
    def get_permissions(self) -> dict:
        return self.permissions
    
    def get_space_key(self) -> int:
        return self.key
        
    def get_name(self) -> str:
        return self.name
    
    def add_document(self, document: list) -> None:
        self.documents.append(document)
    
    def get_documents(self) -> list:
        return self.documents